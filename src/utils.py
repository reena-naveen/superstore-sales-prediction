def evaluate_model(y_true, y_pred):
    """Evaluate the model's predictions against the true values."""
    accuracy = sum(y_true == y_pred) / len(y_true)
    return accuracy

def visualize_results(y_true, y_pred):
    """Visualize the true vs predicted results using matplotlib."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.title('True vs Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics including MSE, MAE."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae}
