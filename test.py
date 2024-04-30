# Define color codes with background equivalents
colors = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'purple': '\033[95m',
    'blue':    '\033[94m',
    'orange': '\033[33m',
    'reset': '\033[0m',  # Reset color to default
    'bg_red': '\033[101m',
    'bg_green': '\033[102m',
    'bg_yellow': '\033[103m',
    'bg_purple': '\033[105m',
    'bg_blue': '\033[104m',
    'bg_orange': '\033[43m'
}

# Example usage:
print(colors['bg_yellow'] + "This is text with a yellow background" + colors['reset'])
print(colors['bg_purple'] + "This is text with a purple background" + colors['reset'])
