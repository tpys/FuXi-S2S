import argparse
import os
import time 
import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort
from copy import deepcopy
from data_util import make_input, print_dataarray

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="FuXi-S2S onnx model file")
parser.add_argument('--input', type=str, required=True, help="The input netcdf data file")
parser.add_argument('--device', type=str, default="cuda", help="The device to run FuXi model")
parser.add_argument('--save_dir', type=str, default="")
parser.add_argument('--total_step', type=int, default=42)
parser.add_argument('--total_member', type=int, default=1)
args = parser.parse_args()


def save_with_progress(ds, save_name, dtype=np.float32):
    from dask.diagnostics import ProgressBar

    if 'time' in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))

    ds = ds.astype(dtype)
    obj = ds.to_netcdf(save_name, compute=False)

    with ProgressBar():
        obj.compute()


def save_like(output, input, member, lead_time, save_dir=""):

    if args.save_dir:
        save_dir = os.path.join(args.save_dir, f"member/{member:02d}")
        os.makedirs(save_dir, exist_ok=True)
        init_time = pd.to_datetime(input.time.data[-1])

        ds = xr.DataArray(
            data=output,
            dims=['time', 'lead_time', 'channel', 'lat', 'lon'],
            coords=dict(
                time=[init_time],
                lead_time=[lead_time],
                channel=input.channel,
                lat=input.lat,
                lon=input.lon,
            )
        ).astype(np.float32)
        print_dataarray(ds)
        save_name = os.path.join(save_dir, f'{lead_time:02d}.nc')
        ds.to_netcdf(save_name)



def load_model(model_name, device):
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    
    if device == "cuda":
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy':'kSameAsRequested'})]
    elif device == "cpu":
        providers=['CPUExecutionProvider']
        options.intra_op_num_threads = 24
    else:
        raise ValueError("device must be cpu or cuda!")

    session = ort.InferenceSession(
        model_name,  
        sess_options=options, 
        providers=providers
    )
    return session


def run_inference(
    model, 
    input, 
    total_step, 
    total_member, 
    save_dir=""
):
    hist_time = pd.to_datetime(input.time.values[-2])
    init_time = pd.to_datetime(input.time.values[-1])
    assert init_time - hist_time == pd.Timedelta(days=1)
    
    lat = input.lat.values 
    lon = input.lon.values 
    batch = input.values[None]
    
    assert lat[0] == 90 and lat[-1] == -90
    print(f'Model initial Time: {init_time.strftime(("%Y%m%d%H"))}')
    print(f"Region: {lat[0]:.2f} ~ {lat[-1]:.2f}, {lon[0]:.2f} ~ {lon[-1]:.2f}")

    for member in range(total_member):
        print(f'Inference member {member:02d} ...')
        new_input = deepcopy(batch)

        start = time.perf_counter()
        for step in range(total_step):
            lead_time = (step + 1)

            inputs = {'input': new_input}        

            if "step" in input_names:
                inputs['step'] = np.array([step], dtype=np.float32)

            if "doy" in input_names:
                valid_time = init_time + pd.Timedelta(days=step)
                doy = min(365, valid_time.day_of_year)/365 
                inputs['doy'] = np.array([doy], dtype=np.float32)

            istart = time.perf_counter()
            new_input, = model.run(None, inputs)
            output = deepcopy(new_input[:, -1:])
            step_time = time.perf_counter() - istart

            print(f"member: {member:02d}, step {step+1:02d}, step_time: {step_time:.3f} sec")
            save_like(output, input, member, lead_time, save_dir)
            
            if step > total_step:
                break

        run_time = time.perf_counter() - start
        print(f'Inference member done, take {run_time:.2f} sec')


def land_to_nan(input, mask, names=['sst']):
    channel = input.channel.data.tolist()
    for ch in names:
        v = input.sel(channel=ch)
        v = v.where(mask)
        idx = channel.index(ch)
        input.data[:, idx] = v.data
    return input



if __name__ == "__main__":
    if os.path.exists(args.input):
        input = xr.open_dataarray(args.input)
    else:
        input = make_input("data/sample")
        input.to_netcdf("data/input.nc")

    mask = xr.open_dataarray("data/mask.nc")
    input = land_to_nan(input, mask)    
    print_dataarray(input)        

    print(f'Load FuXi ...')       
    start = time.perf_counter()
    model = load_model(args.model, args.device)
    input_names = [input.name for input in model.get_inputs()]
    print(f'Load FuXi take {time.perf_counter() - start:.2f} sec')

    run_inference(
        model, 
        input, 
        args.total_step, 
        args.total_member,  
        save_dir=args.save_dir
    )