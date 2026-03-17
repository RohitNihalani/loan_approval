import os 
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, f1_score

class ModelTrainer:

    def TrainModel(self, X, y, preprocessor):
        # 1. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y # 'stratify' ensures balanced splits
        )

        # 2. Define the Pipeline
        # We use class_weight='balanced' to handle the "always rejected" bias
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        # 3. Define Hyperparameter Grid
        # Note the 'model__' prefix to tell the pipeline these belong to the classifier
        param_grid = {
            'model__n_estimators': [100,150, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        }

        # 4. Initialize Grid Search
        # We optimize for 'f1' instead of 'accuracy'
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='f1', 
            n_jobs=-1
        )

        print("Tuning hyperparameters... this may take a minute.")
        grid_search.fit(X_train, y_train)

        # 5. Evaluate the best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print(f"Best Parameters: {grid_search.best_params_}")
        print("\n--- Detailed Performance ---")
        print(classification_report(y_test, y_pred)) # Shows Precision and Recall for both classes

        # 6. Save the best version
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/model.pkl")
        print("Optimized model saved successfully!")

        return best_model