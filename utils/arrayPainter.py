# Define color codes
colors = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'purple': '\033[95m',
    'blue':    '\033[34m',
    'orange': '\033[33m',  # Orange color
    'reset': '\033[0m'  # Reset color to default
}

def draw_array(array,color='reset'):
    length = array.shape[0]
    if len(array.shape) == 1:
        height = 1
    else:
        height = array.shape[1]

    shape_0 = length
    shape_1 = height

    if length > 10 or height > 10:
        if length > height:
            length = 9
            if height > 10:
                height = 7
            else:
                height = height
        else:
            if length > 10:
                length = 7
            else:
                length = length + 2 
            height = 9

    distancer = "   "
    filler = " . "

    viz_matrix = []
    half_h = int(height/2)

    if length % 2 == 0:
        length -= 1
    half_l = int(length/2)

    viz_matrix.append(f"{colors[color]}⎡"+distancer*length+f"⎤{colors['reset']}")
    for h in range(height):
        if h == half_h:
            if len(f"{shape_0}x{shape_1}") == 4:
                viz_matrix.append(f"{colors[color]}⎢"+filler*half_l+f"{shape_0}x{shape_1}  "+filler*(half_l-1)+f"⎥{colors['reset']}")            
            elif len(f"{shape_0}x{shape_1}") == 5:
                viz_matrix.append(f"{colors[color]}⎢"+filler*half_l+f"{shape_0}x{shape_1} "+filler*(half_l-1)+f"⎥{colors['reset']}")            
            else:
                viz_matrix.append(f"{colors[color]}⎢"+filler*half_l+f"{shape_0}x{shape_1}"+filler*half_l+f"⎥{colors['reset']}")
        else:
            viz_matrix.append(f"{colors[color]}⎢"+filler*length+f"⎥{colors['reset']}")
    viz_matrix.append(f"{colors[color]}⎣"+distancer*length+f"⎦{colors['reset']}")

    return viz_matrix

def draw_with_values(array,color='reset'):
    length = array.shape[0]
    if len(array.shape) == 1:
        height = 1
        array = array.reshape(length,height)
    else:
        height = array.shape[1]

    distancer = "     "
    viz_matrix = []

    viz_matrix.append(f"{colors[color]}⎡"+distancer*length+f"⎤{colors['reset']}")
    for h in range(height):
        line = f"{colors[color]}⎢"
        for v in array[:,h]:
            line += f" {v:.1f} "
        line += f"⎥{colors['reset']}"
        viz_matrix.append(line)
    viz_matrix.append(f"{colors[color]}⎣"+distancer*length+f"⎦{colors['reset']}")

    return viz_matrix


def display_array(arrays,colors):
    to_draw = []
    for array,color in zip(arrays,colors):
        to_draw.append(draw_array(array,color))

    longest = max([len(n) for n in to_draw])
    distancer = " "

    for row_n in range(longest):
        print_payload = ""

        for n,draw in enumerate(to_draw):
            if row_n >= longest - len(draw):
                row_m = row_n - (longest - len(draw))
                print_payload += draw[row_m]
            else:
                # weird magical number but somehow I will survive it 
                print_payload += " " * (len(draw[0]) - 9)

        print(print_payload)

def display_with_values(arrays,colors):
    to_draw = []
    for array,color in zip(arrays,colors):
        to_draw.append(draw_with_values(array,color))

    longest = max([len(n) for n in to_draw])
    distancer = " "

    for row_n in range(longest):
        print_payload = ""

        for n,draw in enumerate(to_draw):
            if row_n >= longest - len(draw):
                row_m = row_n - (longest - len(draw))
                print_payload += draw[row_m]
            else:
                # weird magical number but somehow I will survive it 
                print_payload += " " * (len(draw[0]) - 9)

        print(print_payload)
