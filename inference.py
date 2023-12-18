import argparse
import os
import time 
import numpy as np
import xarray as xr
import pandas as pd
from copy import deepcopy
import torch 

from data_util import inv_normalize, print_dataarray


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="The FuXi-S2S model")
parser.add_argument('--input', type=str, required=True, help="The input data")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--save_dir', type=str, default="", help="Where to save the forecasts")
parser.add_argument('--total_step', type=int, default=42)
parser.add_argument('--total_member', type=int, default=51)
args = parser.parse_args()


def save_like(output, template, step, member):

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        mean = xr.open_dataarray("data/mean.nc")
        std = xr.open_dataarray("data/std.nc")
        
        init_time = pd.to_datetime(template.time.data[-1])
        ds = xr.DataArray(
            data=output[None],
            dims=['time', 'step', 'member', 'level', 'lat', 'lon'],
            coords=dict(
                time=[init_time],
                step=[step],
                member=[member],
                level=template.level,
                lat=template.lat,
                lon=template.lon,
            )
        ).astype(np.float32)
        ds = inv_normalize(ds, mean, std)
        # print_dataarray(ds)
        save_name = os.path.join(args.save_dir, f'm{member:02d}_s{step:02d}.nc')
        ds.to_netcdf(save_name)




def run_inference(model, input_nc, total_step, total_member):
    hist_time = pd.to_datetime(input_nc.time.values[-2])
    init_time = pd.to_datetime(input_nc.time.values[-1])
    assert init_time - hist_time == pd.Timedelta(days=1)
    lat = input_nc.lat.values 
    lon = input_nc.lon.values 
    assert lat[0] == 90 and lat[-1] == -90

    input = input_nc.data[None]
    print(f'Model initial Time: {init_time.strftime(("%Y%m%d%H"))}')
    print(f"Region: {lat[0]:.2f} ~ {lat[-1]:.2f}, {lon[0]:.2f} ~ {lon[-1]:.2f}")
    print(f"Input: {input.shape}")

    
    print(f'Inference ...')
    for m in range(total_member):
        curr_input = deepcopy(input)

        for t in range(total_step):
            curr_input = torch.from_numpy(curr_input).to(args.device)
            step = curr_input.new_full((1,), t)
            curr_input = model(curr_input, step).detach().cpu().numpy()
            print(f"member: {m:02d}, step: {t+1:02d}")

            save_like(
                output=curr_input[:, -1:], 
                template=input_nc, 
                step=t+1, 
                member=m,
            )

            if t > total_step:
                break
    
    run_time = time.perf_counter() - start
    print(f'Inference done take {run_time:.2f}')

    
if __name__ == "__main__":
    print(f'Load Input ...')    
    start = time.perf_counter()
    input_nc = xr.open_dataarray(args.input)
    print_dataarray(input_nc)
    print(f'Load Input take {time.perf_counter() - start:.2f} sec')

    print(f'Load FuXi ...')       
    start = time.perf_counter()
    model = torch.jit.load(args.model)
    model = model.to(args.device)
    model.eval()    

    print(f'Load FuXi take {time.perf_counter() - start:.2f} sec')
    run_inference(model, input_nc, args.total_step, args.total_member)