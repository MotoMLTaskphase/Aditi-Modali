import pygame
import random
import math

pygame.init()

FPS = 60

WIDTH, HEIGHT = 400, 400
ROWS = 4
COLS = 4

RECT_HEIGHT = HEIGHT // ROWS
RECT_WIDTH = WIDTH // COLS

OUTLINE_COLOR = (187, 173, 160)
OUTLINE_THICKNESS = 10
BACKGROUND_COLOR = (205, 192, 180)
FONT_COLOR = (119, 110, 101)

FONT = pygame.font.SysFont("comicsans", 60, bold=True)
GAME_OVER_FONT = pygame.font.SysFont("comicsans", 80, bold=True)
MOVE_VEL = 20

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048")


class Tile:
    COLORS = [
        (237, 229, 218),  # 2
        (238, 225, 201),  # 4
        (243, 178, 122),  # 8
        (246, 150, 101),  # 16
        (247, 124, 95),   # 32
        (247, 95, 59),    # 64
        (237, 208, 115),  # 128
        (237, 204, 99),   # 256
        (236, 202, 80),   # 512
    ]

    def __init__(self, value, row, col):
        self.value = value
        self.row = row
        self.col = col
        self.x = col * RECT_WIDTH
        self.y = row * RECT_HEIGHT

    def get_color(self):
        color_index = min(int(math.log2(self.value)) - 1, len(self.COLORS) - 1)
        return self.COLORS[color_index]

    def draw(self, window):
        color = self.get_color()
        pygame.draw.rect(window, color, (self.x, self.y, RECT_WIDTH, RECT_HEIGHT))

        text = FONT.render(str(self.value), 1, FONT_COLOR)
        window.blit(
            text,
            (
                self.x + (RECT_WIDTH / 2 - text.get_width() / 2),
                self.y + (RECT_HEIGHT / 2 - text.get_height() / 2),
            ),
        )

    def set_pos(self, ceil=False):
        if ceil:
            self.row = math.ceil(self.y / RECT_HEIGHT)
            self.col = math.ceil(self.x / RECT_WIDTH)
        else:
            self.row = math.floor(self.y / RECT_HEIGHT)
            self.col = math.floor(self.x / RECT_WIDTH)

    def move(self, delta):
        self.x += delta[0]
        self.y += delta[1]


def draw_grid(window):
    for row in range(1, ROWS):
        y = row * RECT_HEIGHT
        pygame.draw.line(window, OUTLINE_COLOR, (0, y), (WIDTH, y), OUTLINE_THICKNESS)

    for col in range(1, COLS):
        x = col * RECT_WIDTH
        pygame.draw.line(window, OUTLINE_COLOR, (x, 0), (x, HEIGHT), OUTLINE_THICKNESS)

    pygame.draw.rect(window, OUTLINE_COLOR, (0, 0, WIDTH, HEIGHT), OUTLINE_THICKNESS)


def draw_message(window, message):
    # Create semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(128)
    overlay.fill((0, 0, 0))
    window.blit(overlay, (0, 0))

    # Draw main message
    text = GAME_OVER_FONT.render(message, True, (255, 255, 255))
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    window.blit(text, text_rect)

    # Draw instruction
    instruction_font = pygame.font.SysFont("comicsans", 30, bold=True)
    instruction_text = instruction_font.render("Press SPACE to restart or ESC to quit", True, (255, 255, 255))
    instruction_rect = instruction_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    window.blit(instruction_text, instruction_rect)

    pygame.display.update()


def draw(window, tiles, game_state):
    window.fill(BACKGROUND_COLOR)

    for tile in tiles.values():
        tile.draw(window)

    draw_grid(window)

    if game_state in ["won", "lost"]:
        message = "You Won!" if game_state == "won" else "Game Over!"
        draw_message(window, message)
    
    pygame.display.update()


def get_random_pos(tiles):
    available_positions = [
        (row, col) 
        for row in range(ROWS) 
        for col in range(COLS) 
        if f"{row}{col}" not in tiles
    ]
    if not available_positions:
        return None, None
    return random.choice(available_positions)


def check_win_condition(tiles):
    for tile in tiles.values():
        if tile.value >= 2048:
            return True
    return False


def check_possible_moves(tiles):
    # Check for empty spaces
    if len(tiles) < 16:
        return True
    
    # Check for possible merges horizontally and vertically
    for row in range(ROWS):
        for col in range(COLS):
            current_tile = tiles.get(f"{row}{col}")
            if current_tile:
                # Check right
                if col < COLS - 1:
                    next_tile = tiles.get(f"{row}{col+1}")
                    if next_tile and next_tile.value == current_tile.value:
                        return True
                # Check down
                if row < ROWS - 1:
                    next_tile = tiles.get(f"{row+1}{col}")
                    if next_tile and next_tile.value == current_tile.value:
                        return True
    return False


def move_tiles(window, tiles, clock, direction):
    if direction == "left":
        sort_func = lambda x: x.col
        reverse = False
        delta = (-MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col - 1}")
        merge_check = lambda tile, next_tile: tile.x > next_tile.x + MOVE_VEL
        move_check = lambda tile, next_tile: tile.x > next_tile.x + RECT_WIDTH + MOVE_VEL
        ceil = True
    elif direction == "right":
        sort_func = lambda x: x.col
        reverse = True
        delta = (MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == COLS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col + 1}")
        merge_check = lambda tile, next_tile: tile.x < next_tile.x - MOVE_VEL
        move_check = lambda tile, next_tile: tile.x + RECT_WIDTH + MOVE_VEL < next_tile.x
        ceil = False
    elif direction == "up":
        sort_func = lambda x: x.row
        reverse = False
        delta = (0, -MOVE_VEL)
        boundary_check = lambda tile: tile.row == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row - 1}{tile.col}")
        merge_check = lambda tile, next_tile: tile.y > next_tile.y + MOVE_VEL
        move_check = lambda tile, next_tile: tile.y > next_tile.y + RECT_HEIGHT + MOVE_VEL
        ceil = True
    elif direction == "down":
        sort_func = lambda x: x.row
        reverse = True
        delta = (0, MOVE_VEL)
        boundary_check = lambda tile: tile.row == ROWS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row + 1}{tile.col}")
        merge_check = lambda tile, next_tile: tile.y < next_tile.y - MOVE_VEL
        move_check = lambda tile, next_tile: tile.y + RECT_HEIGHT + MOVE_VEL < next_tile.y
        ceil = False

    updated = True
    blocks = set()

    while updated:
        clock.tick(FPS)
        updated = False
        sorted_tiles = sorted(tiles.values(), key=sort_func, reverse=reverse)

        for i, tile in enumerate(sorted_tiles):
            if boundary_check(tile):
                continue

            next_tile = get_next_tile(tile)
            if not next_tile:
                tile.move(delta)
            elif (
                tile.value == next_tile.value
                and tile not in blocks
                and next_tile not in blocks
            ):
                if merge_check(tile, next_tile):
                    tile.move(delta)
                else:
                    next_tile.value *= 2
                    sorted_tiles.pop(i)
                    blocks.add(next_tile)
            elif move_check(tile, next_tile):
                tile.move(delta)
            else:
                continue

            tile.set_pos(ceil)
            updated = True

        update_tiles(window, tiles, sorted_tiles)

    # After move is complete, check win/lose conditions
    if check_win_condition(tiles):
        return "won"
    
    # Add new tile
    row, col = get_random_pos(tiles)
    if row is not None:
        tiles[f"{row}{col}"] = Tile(random.choice([2, 4]), row, col)
    
    # Check if game is lost
    if len(tiles) == 16 and not check_possible_moves(tiles):
        return "lost"
    
    return "continue"


def update_tiles(window, tiles, sorted_tiles):
    tiles.clear()
    for tile in sorted_tiles:
        tiles[f"{tile.row}{tile.col}"] = tile

    draw(window, tiles, "continue")


def generate_tiles():
    tiles = {}
    for _ in range(2):
        row, col = get_random_pos(tiles)
        tiles[f"{row}{col}"] = Tile(2, row, col)
    return tiles


def main(window):
    clock = pygame.time.Clock()
    run = True
    game_state = "continue"
    tiles = generate_tiles()

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

            if event.type == pygame.KEYDOWN:
                if game_state == "continue":
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                        direction = {
                            pygame.K_LEFT: "left",
                            pygame.K_RIGHT: "right",
                            pygame.K_UP: "up",
                            pygame.K_DOWN: "down"
                        }[event.key]
                        new_state = move_tiles(window, tiles, clock, direction)
                        if new_state in ["won", "lost"]:
                            game_state = new_state
                            draw(window, tiles, game_state)
                
                elif game_state in ["won", "lost"]:
                    if event.key == pygame.K_SPACE:
                        # Restart game
                        tiles = generate_tiles()
                        game_state = "continue"
                    elif event.key == pygame.K_ESCAPE:
                        run = False

        # Always draw the current state
        draw(window, tiles, game_state)

    pygame.quit()


if __name__ == "__main__":
    main(WINDOW)