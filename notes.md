**Self-Play RL (PPO/GAE) Notes for Gomoku (Single Network)**

*   **Input Observation:** Board state (player-agnostic, e.g., +1 Black, -1 White, 0 Empty) **plus** an indicator of the current player to move.
*   **Network Output:** The policy (\(\pi\)) and value (\(V\)) output by the neural network are always interpreted **from the perspective of the current player** indicated in the input.
*   **Reward Signal:** The environment provides a reward at the end of the game. It's +1 for the player who just made the winning move (i.e., the reward is always +1 for the *winner*). Intermediate rewards are 0.
*   **GAE & Returns:** Advantage (\(\hat{A}_t\)) and Return (\(R_t = \hat{A}_t + V(s_t)\)) calculations are performed consistently from the perspective of the player whose turn it was at step \(t\). This is ensured by using the modified TD-residual (\(\delta_t = r_{t+1} - \gamma V(s_{t+1}) - V(s_t)\)) which accounts for the value perspective flip between turns.
*   **Actions & Loss:** Actions sampled and loss calculations (using \(R_t\) and \(\hat{A}_t\)) are all relative to the current player at each step.
*   **Result:** This setup trains a single network to effectively play as *both* Black and White, adapting its evaluation and policy based on the current player input. 