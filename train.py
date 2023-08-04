import torch
from torch import optim
from torch.nn import MSELoss
from ActorCriticNetwork import ActorCriticNetwork
from generate_self_play_data import generate_self_play_data

def train():
    # Hyperparameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    model_path = 'model.pth'

    # Create the network and the optimizer
    network = ActorCriticNetwork()

    # Load the model if it exists
    try:
        network.load_state_dict(torch.load(model_path))
        print("Loaded model from disk")
    except FileNotFoundError:
        print("No model found on disk, starting from scratch")

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # Loss functions
    value_loss_fn = MSELoss()
    policy_loss_fn = MSELoss()

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")

        # Generate self-play data
        print("Generating self-play data...")
        game_data = generate_self_play_data(network)
        print(f"Generated {len(game_data)} games")

        # Shuffle the data
        np.random.shuffle(game_data)

        total_loss = 0
        total_value_loss = 0
        total_policy_loss = 0

        for i in range(0, len(game_data), batch_size):
            # Get the batch
            batch = game_data[i:i+batch_size]

            # Separate the batch into states, policies and values
            states, target_policies, target_values = zip(*batch)

            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float32)
            target_policies = torch.tensor(target_policies, dtype=torch.float32)
            target_values = torch.tensor(target_values, dtype=torch.float32)

            # Forward pass
            predicted_policies, predicted_values = network(states)

            # Compute the loss
            value_loss = value_loss_fn(predicted_values, target_values)
            policy_loss = policy_loss_fn(predicted_policies, target_policies)
            loss = value_loss + policy_loss

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} completed")
        print(f"Average loss: {total_loss / len(game_data)}")
        print(f"Average value loss: {total_value_loss / len(game_data)}")
        print(f"Average policy loss: {total_policy_loss / len(game_data)}")

        # Save the model
        torch.save(network.state_dict(), model_path)
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train()
