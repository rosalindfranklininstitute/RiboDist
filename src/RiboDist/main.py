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
from tqdm import tqdm
from glob import glob
import numpy as np

from magicgui import magicgui as mg
from . import func_defs as fd


@mg(
    call_button="Run RiboDist!",
    layout="vertical",
    result_widget=True,

    output_star={"widget_type": "FileEdit",
                 "label": "Path to output star file"},
    output_figs_folder={"widget_type": "FileEdit",
                   "label": "Folder for figures output",
                   "mode": "d"},
    models_folder={"widget_type": "FileEdit",
                   "label": "Folder containing models",
                   "mode": "d"},
    models_format={"widget_type": "FileEdit",
                   "label": f"Filename format for models. \nReplace tilt series number with *. Keep only filename."},
    models_bin={"label": "Binning factor of models",
                "min": 1},
    star_file={"widget_type": "FileEdit",
               "label": "Input star file"},
    star_bin={"label": "Binning factor of star file",
              "min": 1},
)
def rd_main(
        output_star = Path("."),
        output_figs_folder = Path("./figs"),
        models_folder = Path("."),
        models_format = Path("."),
        models_bin = 1,
        star_file = Path("."),
        star_bin = 1,
):
    params = locals()
    full_models_format = str(params['models_folder']) + '/' + str(params['models_format'])
    if not params['output_figs_folder'].is_dir():
        params['output_figs_folder'].mkdir()

    model_TS, model_TS_list = fd.get_model_list(full_models_format)
    ribo_star, TS_list, pixel_size_nm = fd.get_ribo_from_star(star_file=str(params['star_file']))

    thickness_list = []
    skipped_list = []
    tqdm_enum = tqdm(range(len(model_TS_list)))
    factor = pixel_size_nm * params['models_bin'] / params['star_bin']

    for idx in tqdm_enum:
        curr_ts = model_TS_list[idx]
        curr_ts_str = model_TS[idx]
        model_file = re.sub("\*", curr_ts_str, full_models_format)

        TS_matched = True
        try:
            assert(curr_ts in TS_list and glob(model_file))
        except:
            TS_matched = False

        try:
            assert(glob(model_file))
        except:
            skipped_list.append(curr_ts)
            continue

        model = fd.get_model(model_file=model_file)

        #     Segmentation of surfaces
        _, model_upper, model_lower = fd.segment_surfaces(model_in=model)

        #     Plane interpolation
        interped_top, interped_bot, thickness = fd.interpolator(model_upper[:, 1:],
                                                                model_lower[:, 1:],
                                                                100,
                                                                factor)
        thickness_list.append([curr_ts, thickness])

        if TS_matched:
            ribo = fd.get_coords(star_df_in=ribo_star['particles'],
                                 TS=curr_ts,
                                 model_bin=params['models_bin'],
                                 star_bin=params['star_bin'])
            #     Calculate particle-edge distance (if coordinates given)
            to_edge = fd.calc_dist(interped_top, interped_bot, ribo)

            #     Aggregation of data
            df = pd.DataFrame(columns=["x", "y", "z", "to_top", "to_bottom"])
            df.x = ribo[:, 0]
            df.y = ribo[:, 1]
            df.z = ribo[:, 2]
            df.to_top = to_edge[:, 0]
            df.to_bottom = to_edge[:, 1]
            df["to_any_edge"] = df[["to_top", "to_bottom"]].values.min(1)
            df["thickness"] = thickness

            #     Update of star-DataFrame
            ribo_star['particles'].loc[ribo_star['particles'].rlnTS==curr_ts, 'rlnDistToEdge_nm'] = df.to_any_edge.to_numpy() * factor
            ribo_star['particles'].loc[ribo_star['particles'].rlnTS==curr_ts, 'rlnLamellaThickness_nm'] = df.thickness.to_numpy()

            # Saving figures
            fd.savefig(top_in=interped_top,
                       bot_in=interped_bot,
                       ribo_in=ribo,
                       to_edge_in=to_edge,
                       save_path=str(params['output_figs_folder']) + '/' + f"TS_{curr_ts}.png")

            # Write out star file
            sf.write(ribo_star, str(params['output_star']), overwrite=True)

    # Export lamella thickness table
    # thickness_table = ribo_star['particles'][~ribo_star['particles'].rlnDistToEdge_nm.isnull()][['rlnTS', 'rlnLamellaThickness_nm']].sort_values(by='rlnTS').drop_duplicates()
    # thickness_table.to_csv(r'./lamella_thickness_nm.txt', header=None, index=None, sep=' ', mode='a')
    np.savetxt(r'./lamella_thickness_nm.txt',
               thickness_list,
               delimiter = " ",
               fmt="%d %8.4f")

    return f"All finished. {len(skipped_list)} lamellae skipped."



def run_nb():
    rd_main.show(run=True)
