# coding=utf-8
import argparse
from pathlib import Path
import json

def rgb_and_word(token_seq, color_values,
                 left_color_rgb = [240, 128, 128],
                 right_color_rgb = [139,0,0]):
    diff_0  = right_color_rgb[0] - left_color_rgb[0]
    diff_1  = right_color_rgb[1] - left_color_rgb[1]
    diff_2  = right_color_rgb[2] - left_color_rgb[2]
    color_ss, token_ss = "", ""

    for idx, token in enumerate(token_seq):
        if '\\' in token:
            token = token.replace('\\', '\\\\')
        if '#' in token:
            token = token.replace('#', '\#')
        if '$' in token:
            token = token.replace('$', '\$')
        if '&' in token:
            token = token.replace('&', '\&')
        if '_' in token:
            token = token.replace('_', '\_')
        val = color_values[idx]
        r = left_color_rgb[0] + int(val * diff_0)
        g = left_color_rgb[1] + int(val * diff_1)
        b = left_color_rgb[2] + int(val * diff_2)
        c_ss = f"\\definecolor{{color{idx}}}{{RGB}}{{{r}, {g}, {b}}}\n"
        v_ss = f"\\coloredword{{color{idx}}}{{{token}}}\n"

        color_ss += c_ss
        token_ss += v_ss

    return color_ss, token_ss
        

    

    


def hotmap_tex(inst):
    token_seq = inst['token_seq']
    color_values = inst['color_values']

    color_ss, word_ss = rgb_and_word(token_seq, color_values)

    template = r"""
\documentclass{article}
\usepackage{xcolor}
\usepackage{tikz}
""" + color_ss + r"""
% Command to apply a background color to a word with uniform height and alignment
\newcommand{\coloredword}[2]{%
    \tikz[baseline=(X.base)]%
        \node[rectangle, fill=#1, inner sep=2pt, rounded corners, minimum height=1.5em, anchor=base, text depth=0pt, text height=1.2ex] (X) {#2};%
}

\begin{document}""" + word_ss + r"""
\end{document}    
    """
    return template

def main(args):
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for idx, l in enumerate(f):
            inst = json.loads(l)
            ss = hotmap_tex(inst)
            res_path = Path(args.output_dir) / Path(f"hot_map_{idx}.tex")
            res_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(res_path), 'w', encoding='utf-8') as w:
                w.write(ss)
            
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hot map')
    parser.add_argument('--input_path', type=str, help='parser input')
    parser.add_argument('--output_dir', type=str, help='parser output')

    
    args = parser.parse_args() 
    main(args)
    
    pass
