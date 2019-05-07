import subprocess
import os
from xml.etree import ElementTree


SLIC3R_SETTINGS = {
    "executable": ".\slic3r\Slic3r-console.exe",
    "export": "--export-svg",
    "first_layer_height_trigger_string": "--first-layer-height",
    "layer_height_trigger_string": "--layer-height",
    "layer_height": 0.05,
    "output_trigger_string": "--output",
    "output_path": "slic3r_output",
    "svg_layer_tag": "g"
}


def create_svg(filename, settings=SLIC3R_SETTINGS):
    executable = settings["executable"]
    export = settings["export"]
    layer_height_string = ' '.join([settings["layer_height_trigger_string"],
                                   str(settings["layer_height"])])
    first_layer_height_string = ' '.join([settings["first_layer_height_trigger_string"],
                                          str(settings["layer_height"])])
    output_string = ' '.join([settings["output_trigger_string"],
                              settings["output_path"]])

    if not os.path.exists(settings["output_path"]):
        os.mkdir(settings["output_path"])

    command = ' '.join([executable, export, filename, first_layer_height_string, layer_height_string, output_string])
    subprocess.call(command, shell=True)

    return os.path.join(settings["output_path"], filename)


def process_svg(svg_filename):
    with open(svg_filename, 'rt') as f:
        tree = ElementTree.parse(f)

        for group in tree.iterfind('{http://www.w3.org/2000/svg}g'):
            

    for polygon_string in extract_groups(svg_filename):
        process_svg_layer(polygon_string)


def process_svg_layer(polygon_string, output):
    svg = layer_string.encode('utf-8')
    cairosvg.svg2png(svg, write_to=output)


def main():
    create_svg('.\\test_stl\\prism.stl')


if __name__ == "__main__":
    main()