import random
from collections import deque

WIDTH, HEIGHT = 16, 16
NUM_MAZES = 60000

def generate_maze(w, h):
    # Walls between cells:
    #   vertical_walls[y][x] is the wall on the right side of cell (x,y)
    #   horizontal_walls[y][x] is the wall below cell (x,y)
    vertical_walls   = [[True] * w for _ in range(h)]
    horizontal_walls = [[True] * w for _ in range(h)]
    visited = [[False] * w for _ in range(h)]
    
    def carve(x, y):
        visited[y][x] = True
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                # knock down the wall between (x,y) and (nx,ny)
                if dx == 1:      vertical_walls[y][x]   = False
                elif dx == -1:   vertical_walls[y][nx]  = False
                elif dy == 1:    horizontal_walls[y][x] = False
                elif dy == -1:   horizontal_walls[ny][x] = False
                carve(nx, ny)
    
    carve(0, 0)
    return vertical_walls, horizontal_walls

def solve_maze(vwalls, hwalls, start, end):
    w, h = WIDTH, HEIGHT
    queue = deque([start])
    prev = {start: None}
    while queue:
        x, y = queue.popleft()
        if (x, y) == end:
            break
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h): 
                continue
            if (nx, ny) in prev:
                continue
            # check if wall is open
            if dx == 1  and vwalls[y][x]:   continue
            if dx == -1 and vwalls[y][nx]:  continue
            if dy == 1  and hwalls[y][x]:   continue
            if dy == -1 and hwalls[ny][x]:  continue
            prev[(nx, ny)] = (x, y)
            queue.append((nx, ny))
    # reconstruct path
    path = set()
    cur = end
    while cur is not None:
        path.add(cur)
        cur = prev[cur]
    return path

def maze_to_ascii(vwalls, hwalls, start, end, path):
    w, h = WIDTH, HEIGHT
    lines = []
    # top border
    lines.append("+" + ("---+" * w))
    for y in range(h):
        # cell row with vertical walls
        row = "|"
        for x in range(w):
            if (x, y) == start:
                cell = " S "
            elif (x, y) == end:
                cell = " E "
            elif (x, y) in path:
                cell = " * "
            else:
                cell = "   "
            row += cell
            row += " " if not vwalls[y][x] else "|"
        lines.append(row)
        # horizontal walls row
        row = "+"
        for x in range(w):
            row += "   +" if not hwalls[y][x] else "---+"
        lines.append(row)
    return "\n".join(lines)

def random_start_end():
    """Pick two distinct random cells."""
    sx, sy = random.randrange(WIDTH), random.randrange(HEIGHT)
    while True:
        ex, ey = random.randrange(WIDTH), random.randrange(HEIGHT)
        if (ex, ey) != (sx, sy):
            return (sx, sy), (ex, ey)

def main():
    with open("mazes.txt", "w") as f:
        for i in range(NUM_MAZES):
            vwalls, hwalls = generate_maze(WIDTH, HEIGHT)
            start, end   = random_start_end()
            path         = solve_maze(vwalls, hwalls, start, end)
            ascii_maze   = maze_to_ascii(vwalls, hwalls, start, end, path)
            f.write(ascii_maze)
            f.write("\n\n")  # blank line between mazes

if __name__ == "__main__":
    main()
