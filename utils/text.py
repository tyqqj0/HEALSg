# -*- CODING: UTF-8 -*-
# @time 2024/1/25 12:02
# @Author tyqqj
# @File text.py
# @
# @Aim 


def text_in_box(text, length=65, center=True, print_box=True, color='cyan'):
    # Split the text into lines that are at most `length` characters long
    lines = [text[i:i + length] for i in range(0, len(text), length)]

    # Create the box border, with a width of `length` characters
    up_border = '┏' + '━' * (length + 2) + '┓'
    down_border = '┗' + '━' * (length + 2) + '┛'
    # Create the box contents
    contents = '\n'.join(['┃ ' + (line.center(length) if center else line.ljust(length)) + ' ┃' for line in lines])

    # Combine the border and contents to create the final box
    box = '\n'.join([up_border, contents, down_border])

    if print_box:
        ColorPrinter.print_color(box, 'white')

    return box


class ColorPrinter:
    COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }

    @classmethod
    def print_color(cls, text, color):
        if color not in cls.COLORS:
            raise ValueError(f"Unsupported color: {color}")
        print(f"{cls.COLORS[color]}{text}{cls.COLORS['reset']}")


# 分割线
def split_line(length=65):
    return '\n' + '━' * length + '\n'

