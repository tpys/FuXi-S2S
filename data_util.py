import os
import numpy as np
import xarray as xr

__all__ = ["make_input", "print_dataarray"]

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


def print_dataarray(ds, msg='', n=10):
    tid = np.arange(0, ds.shape[0])
    tid = np.append(tid[:n], tid[-n:])    
    v = ds.isel(time=tid)
    msg += f"short_name: {ds.name}, shape: {ds.shape}, value: {v.values.min():.3f} ~ {v.values.max():.3f}"
    
    if 'lat' in ds.dims:
        lat = ds.lat.values
        msg += f", lat: {lat[0]:.3f} ~ {lat[-1]:.3f}"
    if 'lon' in ds.dims:
        lon = ds.lon.values
        msg += f", lon: {lon[0]:.3f} ~ {lon[-1]:.3f}"   

    if "level" in v.dims and len(v.level) > 1:
        for lvl in v.level.values:
            x = v.sel(level=lvl).values
            msg += f"\nlevel: {lvl:04d}, value: {x.min():.3f} ~ {x.max():.3f}"

    if "channel" in v.dims and len(v.channel) > 1:
        for ch in v.channel.values:
            x = v.sel(channel=ch).values
            msg += f"\nchannel: {ch}, value: {x.min():.3f} ~ {x.max():.3f}"

    print(msg)



def make_input(data_dir):
    ds = []
    channel = []
    for (long_name, short_name) in s2s_names:   
        file_name = os.path.join(data_dir, f"{long_name}.nc")
        v = xr.open_dataarray(file_name)

        if short_name == "tp":
            v = np.clip(v * 1000, 0, 1000)

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
        channel += level

    ds = xr.concat(ds, 'level').rename({"level": "channel"})
    ds = ds.assign_coords(channel=channel)
    return ds




