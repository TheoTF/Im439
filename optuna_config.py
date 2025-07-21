import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import os
from datetime import datetime
from config import DatasetConfig, GlobalConfig
from model import DualTimeSeriesModel, custom_loss
from dataloader import TimeSeriesDataset

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

def objective(self, trial: optuna.trial.Trial):
    
    global_config = self.exp_setup(trial)
    model = DualTimeSeriesModel(global_config)

    window_size = 128
    step_size = 64
    max_time_gap = 3 # seconds

    train_dataset_config = DatasetConfig(self.X_train, window_size, step_size, max_time_gap)
    val_dataset_config = DatasetConfig(self.X_val, window_size, step_size, max_time_gap)
    test_dataset_config = DatasetConfig(self.X_test, window_size, step_size, max_time_gap)

    train_dataset = TimeSeriesDataset(train_dataset_config)
    val_dataset = TimeSeriesDataset(val_dataset_config)
    test_dataset = TimeSeriesDataset(test_dataset_config)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
    
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
    tau = trial.suggest_float('tau', 0.5, 5.0)  # Temperature parameter
    delta = trial.suggest_float('delta', 0.1, 2.0)  # Target distance
    
    # Use the batch size from the dataloader configuration
    batch_size = train_dataloader.batch_size
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Setup scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Setup early stopping and callback
    early_stopper = EarlyStopper(patience=15, min_delta=0.001)
    optuna_callback = OptunaCallback(trial, metric_name='val_loss')
    
    # Training parameters
    num_epochs = 100
    best_val_loss = float('inf')
    
    logger.info(f"Starting trial {trial.number}")
    logger.info(f"Config: lr={learning_rate}, batch_size={batch_size}, weight_decay={weight_decay}, tau={tau}, delta={delta}")
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, batch_data in enumerate(train_dataloader):
                # Assuming your dataloader returns: (data, data_aug, target_t, target_f, target_t                               _aug, target_f_aug)
                # Adjust this based on your actual data structure
                if len(batch_data) == 6:
                    data, data_aug, Zt, Zf, Zt_aug, Zf_aug = batch_data
                    data, data_aug = data.to(device), data_aug.to(device)
                    Zt, Zf = Zt.to(device), Zf.to(device)
                    Zt_aug, Zf_aug = Zt_aug.to(device), Zf_aug.to(device)
                else:
                    # If your dataloader has different structure, adjust accordingly
                    data, targets = batch_data
                    data = data.to(device)
                    # Extract or generate the required representations
                    model_output = model(data)
                    # You'll need to adapt this based on your model's output structure
                    Zt, Zf, Zt_aug, Zf_aug = model_output  # Adjust this line
                
                optimizer.zero_grad()
                
                # If model needs to process the data to get representations
                if len(batch_data) != 6:
                    representations = model(data)
                    # Adapt this based on how your model returns the representations
                    Zt, Zf, Zt_aug, Zf_aug = representations
                
                # Calculate custom loss
                loss = custom_loss(Zt, Zf, Zt_aug, Zf_aug, tau=tau, delta=delta)
                
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
                for batch_data in val_dataloader:
                    # Same data handling logic as training
                    if len(batch_data) == 6:
                        data, data_aug, Zt, Zf, Zt_aug, Zf_aug = batch_data
                        data, data_aug = data.to(device), data_aug.to(device)
                        Zt, Zf = Zt.to(device), Zf.to(device)
                        Zt_aug, Zf_aug = Zt_aug.to(device), Zf_aug.to(device)
                    else:
                        data, targets = batch_data
                        data = data.to(device)
                        representations = model(data)
                        Zt, Zf, Zt_aug, Zf_aug = representations
                    
                    loss = custom_loss(Zt, Zf, Zt_aug, Zf_aug, tau=tau, delta=delta)
                    
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
        
        # Final evaluation on test set
        model.eval()
        test_loss = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch_data in self.X_test:
                # Same data handling logic
                if len(batch_data) == 6:
                    data, data_aug, Zt, Zf, Zt_aug, Zf_aug = batch_data
                    data, data_aug = data.to(device), data_aug.to(device)
                    Zt, Zf = Zt.to(device), Zf.to(device)
                    Zt_aug, Zf_aug = Zt_aug.to(device), Zf_aug.to(device)
                else:
                    data, targets = batch_data
                    data = data.to(device)
                    representations = model(data)
                    Zt, Zf, Zt_aug, Zf_aug = representations
                
                loss = custom_loss(Zt, Zf, Zt_aug, Zf_aug, tau=tau, delta=delta)
                
                if loss.dim() > 0:
                    loss = loss.mean()
                
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        
        # Log final results
        writer.add_scalar('Loss/Test', avg_test_loss, epoch)
        logger.info(f'Final Test Loss: {avg_test_loss:.6f}')
        
        # Save trial results
        trial.set_user_attr('test_loss', avg_test_loss)
        trial.set_user_attr('best_val_loss', best_val_loss)
        
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