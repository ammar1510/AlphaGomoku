import pygame

class GomokuRenderer:
    def __init__(self, board_size, cell_size=40):
        self.board_size = board_size
        self.cell_size = cell_size
        self.board_pixel_size = board_size * cell_size
        # Initialize pygame.
        pygame.init()
        self.screen = pygame.display.set_mode((self.board_pixel_size, self.board_pixel_size))
        pygame.display.set_caption("Gomoku")
        # Define colors.
        self.board_color = (240, 217, 181)  # Wood-tone board color.
        self.grid_color = (0, 0, 0)
        self.black_stone = (0, 0, 0)
        self.white_stone = (255, 255, 255)

    def render_board(self, board):
        """
        Renders the board state using pygame.

        Args:
            board (list of list): A 2D array-like representation of the board.
                                  Expected values are 1 (black), -1 (white), or 0 (empty).
        """
        # Fill background.
        self.screen.fill(self.board_color)
        
        # Draw grid lines.
        for i in range(self.board_size):
            # Horizontal lines.
            start_h = (self.cell_size // 2, self.cell_size // 2 + i * self.cell_size)
            end_h = (self.board_pixel_size - self.cell_size // 2, self.cell_size // 2 + i * self.cell_size)
            pygame.draw.line(self.screen, self.grid_color, start_h, end_h, 1)
            # Vertical lines.
            start_v = (self.cell_size // 2 + i * self.cell_size, self.cell_size // 2)
            end_v = (self.cell_size // 2 + i * self.cell_size, self.board_pixel_size - self.cell_size // 2)
            pygame.draw.line(self.screen, self.grid_color, start_v, end_v, 1)
        
        # Draw stones.
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                center = (self.cell_size // 2 + j * self.cell_size,
                          self.cell_size // 2 + i * self.cell_size)
                if int(cell) == 1:
                    pygame.draw.circle(self.screen, self.black_stone, center, self.cell_size // 2 - 2)
                elif int(cell) == -1:
                    pygame.draw.circle(self.screen, self.white_stone, center, self.cell_size // 2 - 2)
                    pygame.draw.circle(self.screen, self.grid_color, center, self.cell_size // 2 - 2, 1)
        pygame.display.flip()

    def pause(self):
        """
        Pause execution until the user presses Enter.
        """
        import time
        time.sleep(1)

    def process_events(self):
        """
        Process pygame events. Useful for handling window closure events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit() 
                
    def close(self):
        """
        Closes the pygame window.
        """
        pygame.quit() 