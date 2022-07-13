# Copyright 2021 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


import re
import sys
from os.path import exists
from pathlib import Path
import pandas as pd
import starfile as sf

from magicgui import magicgui as mg
from . import func_defs as fd


@mg(
    call_button="Run RiboDist!",
    layout="vertical",
    result_widget=True,

    output_star={"widget_type": "FileEdit",
                 "label": "Path to output star file"},
    models_folder={"widget_type": "FileEdit",
                   "label": "Folder containing models",
                   "mode": "d"},
    models_format={"widget_type": "LineEdit",
                   "label": f"Filename format for models. \nReplace tilt series number with <TS>."},
    models_bin={"label": "Binning factor of models",
                "min": 1},
    star_file={"widget_type": "FileEdit",
               "label": "Input star file"},
    star_bin={"label": "Binning factor of star file",
              "min": 1},
)
def rd_main(
        output_star = Path("."),
        models_folder = Path("."),
        models_format = "",
        models_bin = 1,
        star_file = Path("."),
        star_bin = 1,
):
    params = locals()
    full_models_format = str(params['models_folder']) + '/' + models_format

    ribo_star, TS_list, pixel_size_nm = fd.get_ribo_from_star(star_file=str(params['star_file']))

    for _, curr_ts in enumerate(TS_list):
        model_file = re.sub("<TS>", str(curr_ts), full_models_format)
        try:
            assert(exists(model_file))
        except:
            print(f"WARNING: {model_file} doesn't exist. TS{curr_ts} skipped.")
            continue

        ribo = fd.get_coords(star_df_in=ribo_star['particles'],
                             TS=curr_ts,
                             model_bin=params['models_bin'],
                             star_bin=params['star_bin'])
        model = fd.get_model(model_file=model_file)

        #     Segmentation of surfaces
        labels, model_upper, model_lower = fd.segment_surfaces(model_in=model)

        #     Plane interpolation
        interped_top, interped_bot, to_edge = fd.interpolator(ribo, model_upper, model_lower, 100)

        #     Aggregation of data
        df = pd.DataFrame(columns=["x", "y", "z", "to_top", "to_bottom"])
        df.x = ribo[:, 0]
        df.y = ribo[:, 1]
        df.z = ribo[:, 2]
        df.to_top = to_edge[:, 0]
        df.to_bottom = to_edge[:, 1]
        df["to_any_edge"] = df[["to_top", "to_bottom"]].values.min(1)

        #     Update of star-DataFrame
        ribo_star['particles'].loc[ribo_star['particles'].rlnTS==curr_ts, 'rlnDistToEdge_nm'] = df.to_any_edge.to_numpy() * pixel_size_nm * params['models_bin'] / params['star_bin']

    # Write out star file
    sf.write(ribo_star, str(params['output_star']), overwrite=True)

    return "All finished."



def run_nb():
    rd_main.show(run=True)
