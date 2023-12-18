import os
import numpy as np
import xarray as xr

__all__ = ["make_input", "normalize", "inv_normalize", "print_dataarray"]

levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


s2s_names = [
    # pl 
    ('geopotential', 'z'),
    ('temperature', 't'),
    ('u_component_of_wind', 'u'),
    ('v_component_of_wind', 'v'),
    ('specific_humidity', 'q'),

    # sfc 
    ('2m_temperature', 't2m'),
    ('2m_dewpoint_temperature', 'd2m'),
    ('sea_surface_temperature', 'sst'),
    ('top_net_thermal_radiation','ttr'),
    ('10m_u_component_of_wind', '10u'),
    ('10m_v_component_of_wind', '10v'),
    ('100m_u_component_of_wind', '100u'),
    ('100m_v_component_of_wind', '100v'),
    ('mean_sea_level_pressure', 'msl'),
    ('total_column_water_vapour', 'tcwv'),
    ('total_precipitation', 'tp'),
]


def normalize(ds, mean, std):
    ds = (ds - mean) / (std + 1e-12) 
    ds = ds.fillna(0)
    return ds


def inv_normalize(ds, mean, std):
    ds = ds * std + mean 
    tp = ds.sel(level="tp")
    tp = np.exp(tp.clip(0, 7)) - 1
    ds.loc[dict(level="tp")] = tp
    return ds


def print_dataarray(
    ds, 
    topk=10,     
    var_names=["tp", "t2m", "ttr", "z500", "u200", "u850"], 
):
    tid = np.arange(0, ds.shape[0])
    tid = np.append(tid[:topk], tid[-topk:])        
    info = f"name: {ds.name}, shape: {ds.shape}"
    
    if 'lat' in ds.dims:
        lat = ds.lat.values
        info += f", lat: {lat[0]:.3f} ~ {lat[-1]:.3f}"
    if 'lon' in ds.dims:
        lon = ds.lon.values
        info += f", lon: {lon[0]:.3f} ~ {lon[-1]:.3f}"        

    v = ds.isel(time=tid)
    if "level" in v.dims:
        for lvl in var_names:
            x = v.sel(level=lvl).values
            info += f'\nlevel: {lvl}, value: {x.min():.3f} ~ {x.max():.3f}'
    elif "channel" in v.dims:
        for ch in var_names:
            x = v.sel(channel=ch).values
            info += f'\nchannel: {ch}, value: {x.min():.3f} ~ {x.max():.3f}'
    else:
        info += f', value: {v.values.min():.3f} ~ {v.values.max():.3f}'
    print(info)



def make_input(data_dir):
    ds = []
    for (long_name, short_name) in s2s_names:   
        file_name = os.path.join(data_dir, f"{long_name}.nc")
        v = xr.open_dataarray(file_name)

        if short_name == "tp":
            v = np.log(1 + v * 1000)
        elif short_name == "ttr":
            v = v / 3600
        if v.level.values[0] != 1000:
            v = v.reindex(level=v.level[::-1])

        if short_name in ['z', 't', 'u', 'v', 'q']:
            level = [f'{short_name}{l}' for l in v.level.values]
        else:
            level = [short_name]

        v.name = "data"
        v.attrs = {}        
        v = v.assign_coords(level=level)
        ds.append(v)

    ds = xr.concat(ds, 'level')
    return ds


def test_make_input():
    mean = xr.open_dataarray("data/mean.nc")
    std = xr.open_dataarray("data/std.nc")
    ds = make_input(data_dir="data/sample")
    ds = normalize(ds, mean, std)
    print_dataarray(ds)
    ds.to_netcdf("data/input.nc")




