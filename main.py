import ludopy
from ludopy import player
import numpy as np
from NN_model import *

# Init New model
main_model = NN()
# Load a trained model:
main_model.set_model(keras.models.load_model("model_v1"))



## Evaluated trained agent


def novice_bot(move_pieces):
    # Random move bot
    if len(move_pieces) >= 1:
        action = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        action = -1
    return action

def play_game(num_games):

    # initialize game
    winner = [0, 0, 0, 0]
    g = ludopy.Game()
    there_is_a_winner = False
    reward = 0




    for episode in range(0, num_games):
        BUFFER = [] # Buffer to save an entire run

        # Play one game of ludo
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
             there_is_a_winner), player_i = g.get_observation()

            if player_i == 0:
                if len(move_pieces):
                    state = np.concatenate((dice / 6, player_pieces / 57, enemy_pieces.flatten().T / 57), axis=None).reshape((1, 17))
                    state_list = state.tolist()

                    action = main_model.get_action(state, move_pieces)

                    BUFFER.append((state, action))
                else:
                    action = -1


            else:
                action = novice_bot(move_pieces)

            # check if there is a winner to end game
            _, _, _, _, _, there_is_a_winner = g.answer_observation(action)

        win = g.get_winners_of_game()[0]
        winner[win] = winner[win] + 1
        print(str(win) + " is the winner!")

        if winner[win] == 0:
            reward = 10
        else:
            reward = -10

        main_model.train_model(BUFFER, reward)

        # reset game
        g.reset()
        there_is_a_winner = False



def run_LUDO_AI():

    num_games = 100
    play_game(num_games)
    main_model.main_model.save("model_v1")

if __name__ == '__main__':
    run_LUDO_AI()


