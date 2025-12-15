"""
Drug-Target Binding Affinity Prediction
========================================
A machine learning pipeline for predicting drug-target binding affinity
using molecular and protein features.

Author: Anh Vu Tran
Date: 2025
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


class DrugTargetPredictor:
    """
    A class for drug-target binding affinity prediction using Random Forest.
    
    This predictor:
    1. Generates synthetic drug and target features
    2. Creates drug-target interaction pairs with binding affinity scores
    3. Trains a Random Forest model
    4. Evaluates performance and provides visualizations
    """
    
    def __init__(self, n_drugs=500, n_targets=100, n_interactions=10000, random_state=42):
        """
        Initialize the predictor with dataset parameters.
        """
        self.n_drugs = n_drugs
        self.n_targets = n_targets
        self.n_interactions = n_interactions
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        self.drug_df = None
        self.target_df = None
        self.interaction_df = None
        self.model = None
        self.scaler = StandardScaler()
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
    def generate_drug_features(self):
        """Generate synthetic drug molecular features."""
        print("Generating drug features...")
        
        drug_features = {
            'Drug_ID': [f'DRUG_{i:04d}' for i in range(self.n_drugs)],
            'Molecular_Weight': np.random.normal(350, 100, self.n_drugs),
            'LogP': np.random.normal(2.5, 1.5, self.n_drugs),
            'NumAtoms': np.random.randint(20, 80, self.n_drugs),
            'NumRings': np.random.randint(1, 6, self.n_drugs),
            'Polar_Surface_Area': np.random.normal(70, 30, self.n_drugs),
            'Num_HB_Donors': np.random.randint(0, 5, self.n_drugs),
            'Num_HB_Acceptors': np.random.randint(2, 10, self.n_drugs)
        }
        
        self.drug_df = pd.DataFrame(drug_features)
        print(f" Generated {len(self.drug_df)} drugs with {len(self.drug_df.columns)-1} features")
        
        return self.drug_df
    
    def generate_target_features(self):
        """Generate synthetic protein target features."""
        print("Generating target features...")
        
        target_features = {
            'Target_ID': [f'TARGET_{i:03d}' for i in range(self.n_targets)],
            'Protein_Length': np.random.randint(200, 800, self.n_targets),
            'Hydrophobicity': np.random.normal(0, 1, self.n_targets),
            'Instability_Index': np.random.normal(35, 15, self.n_targets),
            'Isoelectric_Point': np.random.normal(7, 2, self.n_targets),
            'Net_Charge': np.random.randint(-10, 10, self.n_targets),
            'Num_Domains': np.random.randint(1, 5, self.n_targets),
            'Active_Site_Volume': np.random.normal(1500, 400, self.n_targets)
        }
        
        self.target_df = pd.DataFrame(target_features)
        print(f" Generated {len(self.target_df)} targets with {len(self.target_df.columns)-1} features")
        
        return self.target_df
    
    def generate_interactions(self):
        """Generate drug-target interactions with binding affinity scores."""
        print("Generating drug-target interactions...")
        
        drug_indices = np.random.randint(0, self.n_drugs, self.n_interactions)
        target_indices = np.random.randint(0, self.n_targets, self.n_interactions)
        
        interactions = []
        kiba_scores = []
        
        for i, (drug_idx, target_idx) in enumerate(zip(drug_indices, target_indices)):
            drug = self.drug_df.iloc[drug_idx]
            target = self.target_df.iloc[target_idx]
            
            # Calculate binding affinity based on molecular compatibility
            size_factor = 1 / (1 + abs(drug['Molecular_Weight'] - target['Active_Site_Volume']/4))
            hydrophobic_match = 1 / (1 + abs(drug['LogP'] - target['Hydrophobicity']))
            hbond_potential = (drug['Num_HB_Donors'] + drug['Num_HB_Acceptors']) / 10
            complexity_match = 1 / (1 + abs(drug['NumRings'] - target['Num_Domains']))
            
            base_affinity = size_factor + hydrophobic_match + hbond_potential + complexity_match
            noise = np.random.normal(0, 0.5)
            kiba_score = max(0, base_affinity + noise)
            
            interactions.append({
                'Drug_ID': drug['Drug_ID'],
                'Target_ID': target['Target_ID'],
                'Drug_Idx': drug_idx,
                'Target_Idx': target_idx
            })
            kiba_scores.append(kiba_score)
        
        self.interaction_df = pd.DataFrame(interactions)
        self.interaction_df['KIBA_Score'] = kiba_scores
        
        print(f" Generated {len(self.interaction_df)} interactions")
        print(f" KIBA Score range: {min(kiba_scores):.3f} - {max(kiba_scores):.3f}")
        
        return self.interaction_df
    
    def generate_data(self):
        """Generate complete synthetic dataset."""
        print("\n" + "="*60)
        print("STEP 1: DATA GENERATION")
        print("="*60)
        
        self.generate_drug_features()
        self.generate_target_features()
        self.generate_interactions()
        
        print("\nDataset Summary:")
        print(f"  - Drugs: {self.n_drugs}")
        print(f"  - Targets: {self.n_targets}")
        print(f"  - Interactions: {self.n_interactions}")
        
    def prepare_features(self):
        """Prepare combined features for ML training."""
        print("\n" + "="*60)
        print("STEP 2: FEATURE PREPARATION")
        print("="*60)
        
        # Prepare drug features (exclude ID column)
        drug_feature_cols = [col for col in self.drug_df.columns if col != 'Drug_ID']
        
        # Prepare target features (exclude ID column)
        target_feature_cols = [col for col in self.target_df.columns if col != 'Target_ID']
        
        # Combine features for each interaction
        X_drug = self.drug_df[drug_feature_cols].iloc[self.interaction_df['Drug_Idx'].values].values
        X_target = self.target_df[target_feature_cols].iloc[self.interaction_df['Target_Idx'].values].values
        
        X = np.hstack([X_drug, X_target])
        y = self.interaction_df['KIBA_Score'].values
        
        # Store feature names
        self.feature_names = drug_feature_cols + target_feature_cols
        
        print(f" Combined features shape: {X.shape}")
        print(f" Target shape: {y.shape}")
        print(f" Feature names: {self.feature_names}")
        
        return X, y
    
    def train(self, test_size=0.2):
        """Train the Random Forest model."""
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING")
        print("="*60)
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\nData Split:")
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Testing samples: {len(self.X_test)}")
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\nTraining Random Forest model...")
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print(" Model training completed!")
        
        return self.model
    
    def evaluate(self):
        """Evaluate model performance."""
        print("\n" + "="*60)
        print("STEP 4: MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred_train = self.model.predict(self.X_train_scaled)
        y_pred_test = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train': {
                'mse': mean_squared_error(self.y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'mae': mean_absolute_error(self.y_train, y_pred_train),
                'r2': r2_score(self.y_train, y_pred_train)
            },
            'test': {
                'mse': mean_squared_error(self.y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'mae': mean_absolute_error(self.y_test, y_pred_test),
                'r2': r2_score(self.y_test, y_pred_test)
            }
        }
        
        print("\nTraining Set Performance:")
        print(f"  - MSE:  {metrics['train']['mse']:.4f}")
        print(f"  - RMSE: {metrics['train']['rmse']:.4f}")
        print(f"  - MAE:  {metrics['train']['mae']:.4f}")
        print(f"  - R²:   {metrics['train']['r2']:.4f}")
        
        print("\nTest Set Performance:")
        print(f"  - MSE:  {metrics['test']['mse']:.4f}")
        print(f"  - RMSE: {metrics['test']['rmse']:.4f}")
        print(f"  - MAE:  {metrics['test']['mae']:.4f}")
        print(f"  - R²:   {metrics['test']['r2']:.4f}")
        
        self.y_pred_test = y_pred_test
        self.metrics = metrics
        
        return metrics
    
    def get_feature_importance(self):
        """Get and display feature importance."""
        print("\n" + "="*60)
        print("STEP 5: FEATURE IMPORTANCE")
        print("="*60)
        
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['Feature']:25s}: {row['Importance']:.4f}")
        
        self.importance_df = importance_df
        
        return importance_df
    
    def plot_results(self, save=True):
        """Generate and save visualization plots."""
        print("\n" + "="*60)
        print("STEP 6: VISUALIZATION")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Drug-Target Binding Affinity Prediction Results', fontsize=16, y=1.02)
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(self.y_test, self.y_pred_test, alpha=0.5, s=10)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual KIBA Score')
        axes[0, 0].set_ylabel('Predicted KIBA Score')
        axes[0, 0].set_title(f'Actual vs Predicted (R² = {self.metrics["test"]["r2"]:.3f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance
        top_features = self.importance_df.head(10)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        axes[0, 1].barh(top_features['Feature'], top_features['Importance'], color=colors)
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Top 10 Feature Importance')
        axes[0, 1].invert_yaxis()
        
        # 3. Residual Distribution
        residuals = self.y_test - self.y_pred_test
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residual (Actual - Predicted)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. KIBA Score Distribution
        axes[1, 1].hist(self.interaction_df['KIBA_Score'], bins=50, alpha=0.7, 
                        color='lightcoral', edgecolor='black')
        axes[1, 1].set_xlabel('KIBA Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('KIBA Score Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('output/prediction_results.png', dpi=300, bbox_inches='tight')
            print(" Saved: output/prediction_results.png")
        
        plt.show()
        
    def predict_single(self, drug_features, target_features):
        """Predict binding affinity for a single drug-target pair."""
        combined = np.array(drug_features + target_features).reshape(1, -1)
        combined_scaled = self.scaler.transform(combined)
        prediction = self.model.predict(combined_scaled)[0]
        
        return prediction
    
    def run_pipeline(self):
        """Run the complete prediction pipeline."""
        print("\n" + "="*70)
        print("     DRUG-TARGET BINDING AFFINITY PREDICTION PIPELINE")
        print("="*70)
        
        self.generate_data()
        self.train()
        self.evaluate()
        self.get_feature_importance()
        self.plot_results()
        
        print("\n" + "="*70)
        print("     PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return self.metrics


def main():
    """Main function to run the prediction pipeline."""
    # Initialize predictor
    predictor = DrugTargetPredictor(
        n_drugs=500,
        n_targets=100,
        n_interactions=10000,
        random_state=42
    )
    
    # Run complete pipeline
    metrics = predictor.run_pipeline()
    
    # Example: Predict for a new drug-target pair
    print("\n" + "="*60)
    print("EXAMPLE: Single Prediction")
    print("="*60)
    
    # Example drug features
    drug_features = [350, 2.5, 45, 3, 70, 2, 5]
    # Example target features  
    target_features = [450, 0.1, 35, 7, 2, 3, 1500]
    
    prediction = predictor.predict_single(drug_features, target_features)
    print(f"\nDrug features: {drug_features}")
    print(f"Target features: {target_features}")
    print(f"Predicted KIBA Score: {prediction:.3f}")
    
    return predictor


if __name__ == "__main__":
    predictor = main()
