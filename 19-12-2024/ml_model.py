import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import copy
import pygame
from game_2048 import main, WINDOW, generate_tiles, move_tiles

class RandomForest2048Agent:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=10,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.training_data = []
        self.training_labels = []
        
    def extract_features(self, tiles):
        """Extract meaningful features from the game state"""
        # Initialize the 4x4 grid
        grid = np.zeros((4, 4))
        for key, tile in tiles.items():
            row, col = int(key[0]), int(key[1])
            grid[row][col] = tile.value
            
        features = []
        
        features.extend(grid.flatten())
        
        features.append(np.sum(grid))
        
        features.append(len(np.where(grid == 0)[0]))
        
        features.append(np.max(grid))
        
        for i in range(4):
  
            diff_h = np.diff(grid[i])
            features.append(np.sum(diff_h > 0))
            features.append(np.sum(diff_h < 0))
            
            diff_v = np.diff(grid[:, i])
            features.append(np.sum(diff_v > 0))
            features.append(np.sum(diff_v < 0))
        

        mergeable_h = np.sum(grid[:, :-1] == grid[:, 1:])
        mergeable_v = np.sum(grid[:-1, :] == grid[1:, :])
        features.extend([mergeable_h, mergeable_v])
        
        features.extend([
            grid[0, 0], grid[0, 3],
            grid[3, 0], grid[3, 3]
        ])
        
        return np.array(features)
    
    def get_valid_moves(self, tiles):
        """Get list of valid moves in current state"""
        valid_moves = []
        original_tiles = copy.deepcopy(tiles)
        
        for direction in ["up", "down", "left", "right"]:
            test_tiles = copy.deepcopy(original_tiles)
            if self.is_valid_move(test_tiles, direction):
                valid_moves.append(direction)
        
        return valid_moves
    
    def is_valid_move(self, tiles, direction):
        """Check if a move is valid by seeing if it changes the board state"""
        original_state = self.get_board_state(tiles)
        move_tiles(WINDOW, tiles, pygame.time.Clock(), direction)
        new_state = self.get_board_state(tiles)
        return not np.array_equal(original_state, new_state)
    
    def get_board_state(self, tiles):
        """Get the current board state as a numpy array"""
        state = np.zeros((4, 4))
        for key, tile in tiles.items():
            row, col = int(key[0]), int(key[1])
            state[row][col] = tile.value
        return state
    
    def calculate_move_score(self, old_tiles, new_tiles):
        """Calculate the score/quality of a move"""
        old_score = sum(tile.value for tile in old_tiles.values())
        new_score = sum(tile.value for tile in new_tiles.values())
        score_diff = new_score - old_score
        
        empty_cells_bonus = (16 - len(new_tiles)) * 10  
        merge_bonus = 0
        if len(new_tiles) < len(old_tiles):
            merge_bonus = 50  
        return score_diff + empty_cells_bonus + merge_bonus
    
    def collect_training_data(self, num_games=100):
        """Play games and collect training data"""
        print("Collecting training data...")
        
        for game in range(num_games):
            tiles = generate_tiles()
            moves_made = 0
            game_score = 0
            
            while True:
                valid_moves = self.get_valid_moves(tiles)
                if not valid_moves:
                    break
                    
                current_features = self.extract_features(tiles)
                best_move = None
                best_score = float('-inf')
                

                for move in valid_moves:
                    test_tiles = copy.deepcopy(tiles)
                    move_tiles(WINDOW, test_tiles, pygame.time.Clock(), move)
                    score = self.calculate_move_score(tiles, test_tiles)
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                
                self.training_data.append(current_features)
                self.training_labels.append(
                    ["up", "down", "left", "right"].index(best_move)
                )
                
                move_tiles(WINDOW, tiles, pygame.time.Clock(), best_move)
                moves_made += 1
                game_score += best_score
                

                if moves_made > 1000:
                    break
            
            if (game + 1) % 10 == 0:
                print(f"Completed {game + 1} games. Last game score: {game_score}")
    
    def train(self):
        """Train the random forest model"""
        print("Training random forest model...")
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        self.model.fit(X, y)
        print("Training completed!")
    
    def select_move(self, tiles):
        """Select the best move for the current state"""
        features = self.extract_features(tiles)
        valid_moves = self.get_valid_moves(tiles)
        
        if not valid_moves:
            return None
            

        move_probs = self.model.predict_proba([features])[0]
        move_options = ["up", "down", "left", "right"]
        

        valid_probs = [
            (move_options[i], prob) 
            for i, prob in enumerate(move_probs) 
            if move_options[i] in valid_moves
        ]
        
        best_move = max(valid_probs, key=lambda x: x[1])[0]
        return best_move
    
    def save_model(self, filename='2048_rf_model.pkl'):
        """Save the trained model"""
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='2048_rf_model.pkl'):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filename}")


def evaluate_agent(agent, num_games=10):
    """Evaluate the agent's performance"""
    scores = []
    max_tiles = []
    
    for game in range(num_games):
        tiles = generate_tiles()
        moves = 0
        game_score = 0
        
        while True:
            move = agent.select_move(tiles)
            if not move:
                break
                
            old_tiles = copy.deepcopy(tiles)
            move_tiles(WINDOW, tiles, pygame.time.Clock(), move)
            game_score += agent.calculate_move_score(old_tiles, tiles)
            moves += 1
            
            if moves > 1000:  
                break
        
        score = sum(tile.value for tile in tiles.values())
        max_tile = max(tile.value for tile in tiles.values())
        
        scores.append(score)
        max_tiles.append(max_tile)
        
        print(f"Game {game + 1}: Score = {score}, Max Tile = {max_tile}, Moves = {moves}")
    
    print(f"\nEvaluation Results:")
    print(f"Average Score: {sum(scores) / len(scores):.2f}")
    print(f"Average Max Tile: {sum(max_tiles) / len(max_tiles):.2f}")
    print(f"Highest Score: {max(scores)}")
    print(f"Highest Tile Achieved: {max(max_tiles)}")


def main():
    """Main training and evaluation function"""

    pygame.init()
    
    agent = RandomForest2048Agent()
    
    try:

        print("Starting training phase...")
        agent.collect_training_data(num_games=100)
        agent.train()
        agent.save_model()

        print("\nStarting evaluation phase...")
        evaluate_agent(agent, num_games=10)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
        agent.save_model('2048_rf_model_interrupted.pkl')
        print("Model saved!")
    
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()