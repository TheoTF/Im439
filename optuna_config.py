import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import os
from datetime import datetime
from config import DatasetConfig, GlobalConfig, AugmentationConfig, TimeEncoderConfig, FreqEncoderConfig
from model import DualTimeSeriesModel, custom_loss
from dataloader import TimeSeriesDataset
from transformations import CreateInputs

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class OptunaCallback:
    def __init__(self, trial, metric_name='val_loss'):
        self.trial = trial
        self.metric_name = metric_name
        
    def __call__(self, epoch, metrics):
        # Report intermediate value to Optuna
        if self.metric_name in metrics:
            self.trial.report(metrics[self.metric_name], epoch)
            
        # Check if trial should be pruned
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

class OptunaOptimizer():
    def __init__(self, X_train, X_val_normal, X_val_abnormal, X_test, exp_name, db_name=None, 
                 db_user=None, db_password=None, db_host='localhost', db_port=5432, sql_type='postgres'):

        self.X_train = X_train
        self.X_val_normal = X_val_normal
        self.X_val_abnormal = X_val_abnormal
        self.X_test = X_test
        self.exp_name = exp_name

        # Uncomment the following lines if you want to use a database for Optuna
        # if sql_type == 'postgres':
        #     self.db_host = db_host
        #     self.db_name = db_name
        #     self.db_user = db_user
        #     self.db_password = db_password
        #     if db_user is None or db_password is None:
        #         raise ValueError("For PostgreSQL, db_user and db_password must be provided.")
        #     self.db_conn = psycopg2.connect(host= self.db_host, 
        #                                     database=self.db_name, 
        #                                     user=self.db_user, 
        #                                     password=self.db_password, 
        #                                     port=db_port)
        #     self.db_conn.close()

        # else:
        #     print("Warning: Using SQLite for Optuna. SQLite is not recommended because it can cause issues with parallel trials.")
        #     self.db_conn = sqlite3.connect(db_name)
        #     self.db_conn.close()
        
    
    def exp_setup(self, trial: optuna.trial.Trial):
        """
        Sets up the experiment by defining the transformations hyperparameters to optimize.
        """

        aug_config = AugmentationConfig(
            jitter_sigma=trial.suggest_float("jitter_sigma", low=0.0, high=5.0, step=0.5),
            scaling_sigma=trial.suggest_float("scaling_sigma", low=1.0, high=5.0, step=0.5),
            remove_segment=trial.suggest_int("remove_segment", low=1, high=4, step=1),
            remove_n_signals=trial.suggest_int("remove_n_signals", low=0, high=8, step=1),
            add_segment=trial.suggest_int("add_segment", low=1, high=4, step=1),
            add_n_peaks=trial.suggest_int("add_n_peaks", low=1, high=8, step=1),
            add_mean_peak=trial.suggest_float("add_mean_peak", low=30.0, high=80.0, step=10.0),
            add_rand_error=trial.suggest_float("add_rand_error", low=5.0, high=20.0, step=5.0)
        )

        global_config = GlobalConfig(
            augmentation_config=aug_config,
        )

        return global_config


    def objective(self, trial: optuna.trial.Trial):

        global_config = self.exp_setup(trial)
        model = DualTimeSeriesModel(global_config)

        train_dataloader = DataLoader(self.X_train, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
        val_dataloader_normal = DataLoader(self.X_val_normal, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
        val_dataloader_abnormal = DataLoader(self.X_val_abnormal, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
        test_dataloader = DataLoader(self.X_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

        input_creation = CreateInputs(global_config)
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Setup logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/trial_{trial.number}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup TensorBoard writer
        writer = SummaryWriter(log_dir)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Hyperparameters (suggest these in your exp_setup method)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        delta = trial.suggest_float('delta', 0.1, 2.0)  # Target distance
        
        # Use the batch size from the dataloader configuration
        batch_size = train_dataloader.batch_size
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup scheduler (optional)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5,
        )
        
        # Setup early stopping and callback
        early_stopper = EarlyStopper(patience=15, min_delta=0.001)
        optuna_callback = OptunaCallback(trial, metric_name='val_loss')
        
        # Training parameters
        num_epochs = 10
        best_val_loss = float('inf')
        
        logger.info(f"Starting trial {trial.number}")
        logger.info(f"Config: lr={learning_rate}, batch_size={batch_size}, weight_decay={weight_decay}, delta={delta}")
        logger.info(f"Device: {device}")

        try:
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_batches = 0
                
                for batch_idx, batch_data in enumerate(train_dataloader):
                    # Assuming your dataloader returns: (data, data_aug, target_t, target_f, target_t                               _aug, target_f_aug)
                    # Adjust this based on your actual data structure
                    x_time_original, x_time_aug, x_freq_original, x_freq_aug = input_creation(batch_data)
                    x_time_original, x_time_aug = x_time_original.to(device), x_time_aug.to(device)
                    x_freq_original, x_freq_aug = x_freq_original.to(device), x_freq_aug.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    Zt, Zf = model(x_time_original, x_freq_original)
                    Zt_aug, Zf_aug = model(x_time_aug, x_freq_aug)
                    
                    # Calculate custom loss
                    loss = custom_loss(Zt, Zf, Zt_aug, Zf_aug, delta=delta)

                    # Handle batch dimension if loss is per-sample
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    # Log batch-level metrics occasionally
                    if batch_idx % 100 == 0:
                        logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
                
                avg_train_loss = train_loss / train_batches
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_data_normal, batch_data_abnormal in zip(val_dataloader_normal, val_dataloader_abnormal):
                        # Same data handling logic as training
                        x_time_normal, x_freq_normal = input_creation(batch_data_normal, augment=False)
                        x_time_abnormal, x_freq_abnormal = input_creation(batch_data_abnormal, augment=False)
                        x_time_original, x_time_aug = x_time_normal.to(device), x_time_abnormal.to(device)
                        x_freq_original, x_freq_aug = x_freq_normal.to(device), x_freq_abnormal.to(device)

                        Zt, Zf = model(x_time_original, x_freq_original)
                        Zt_aug, Zf_aug = model(x_time_aug, x_freq_aug)

                        loss = custom_loss(Zt, Zf, Zt_aug, Zf_aug, delta=delta)
                        
                        if loss.dim() > 0:
                            loss = loss.mean()
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                
                # Update learning rate scheduler
                scheduler.step(avg_val_loss)
                
                # Log metrics
                metrics = {
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                
                # Log to TensorBoard
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                
                # Log to console
                logger.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
                
                # Optuna callback
                optuna_callback(epoch, metrics)
                
                # Early stopping check
                if early_stopper(avg_val_loss, model):
                    logger.info(f'Early stopping at epoch {epoch}')
                    break
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), f"{log_dir}/best_model.pth")
            
            # # Final evaluation on test set
            # model.eval()
            # test_loss = 0.0
            # test_batches = 0
            
            # # with torch.no_grad():
            # #     for batch_data in test_dataloader:
            # #         x_time_original, x_time_aug, x_freq_original, x_freq_aug = input_creation(batch_data)
            # #         x_time_original, x_time_aug = x_time_original.to(device), x_time_aug.to(device)
            # #         x_freq_original, x_freq_aug = x_freq_original.to(device), x_freq_aug.to(device)

            # #         Zt, Zf = model(x_time_original, x_freq_original)
            # #         Zt_aug, Zf_aug = model(x_time_aug, x_freq_aug)

            # #         loss = custom_loss(Zt, Zf, Zt_aug, Zf_aug, delta=delta)

            # #         if loss.dim() > 0:
            # #             loss = loss.mean()
                    
            # #         test_loss += loss.item()
            # #         test_batches += 1
            
            # avg_test_loss = test_loss / test_batches
            
            # # Log final results
            # writer.add_scalar('Loss/Test', avg_test_loss, epoch)
            # logger.info(f'Final Test Loss: {avg_test_loss:.6f}')
            
            # # Save trial results
            # trial.set_user_attr('test_loss', avg_test_loss)
            # trial.set_user_attr('best_val_loss', best_val_loss)
            
            # Close writer
            writer.close()
            
            # Return the metric to optimize (typically validation loss)
            return best_val_loss
            
        except optuna.exceptions.TrialPruned:
            logger.info(f"Trial {trial.number} pruned")
            writer.close()
            raise
        
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            writer.close()
            raise

