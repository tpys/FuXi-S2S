import os
import numpy as np
import pandas as pd
import xarray as xr
import pygrib as pg

__all__ = ["make_input", "print_dataarray"]

unit_scale = dict(gh=9.80665, tp=1000)
levels_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

cra_names = dict(
    z=dict(prefix='GPH', short_name='gh', long_name="geopotential", levels=levels_13),
    t=dict(prefix='TEM', short_name='t', long_name="temperature", levels=levels_13),
    u=dict(prefix='WIU', short_name='u', long_name="u_component_of_wind", levels=levels_13),
    v=dict(prefix='WIV', short_name='v', long_name="v_component_of_wind", levels=levels_13),
    q=dict(prefix='SHU', short_name='q', long_name="specific_humidity", levels=levels_13),
    t2m=dict(prefix='SURFACE', short_name='2t', long_name="2m_temperature", levels=[1]),
    d2m=dict(prefix='SINGLEA', short_name='2d', long_name="2m_dewpoint_temperature", levels=[1]),
    sst=dict(prefix='SST', short_name='sst', long_name="sea_surface_temperature", levels=[1]),
    ttr=dict(prefix='OLR', short_name='olr', long_name="top_net_thermal_radiation", levels=[1]),
    u10m=dict(prefix='SINGLEA', short_name='10u', long_name="10m_u_component_of_wind", levels=[1]),
    v10m=dict(prefix='SINGLEA', short_name='10v', long_name="10m_v_component_of_wind", levels=[1]),
    u100m=dict(prefix='SINGLE', short_name='100u', long_name="100m_u_component_of_wind", levels=[1]),
    v100m=dict(prefix='SINGLE', short_name='100v', long_name="100m_v_component_of_wind", levels=[1]),    
    msl=dict(prefix='SINGLE', short_name='prmsl', long_name="mean_sea_level_pressure", levels=[1]),    
    tcwv=dict(prefix='SINGLE', short_name='pwat', long_name="total_column_water_vapour", levels=[1]),    
    tp=dict(prefix='SINGLE', short_name='pwat', long_name="total_precipitation", levels=[1]),    
)


def get_file_name(data_dir, prefix):
    for file_name in os.listdir(data_dir):
        if f"_{prefix}_" in file_name:
            return os.path.join(data_dir, file_name)
        elif f"-{prefix}-" in file_name:
            return os.path.join(data_dir, file_name)
    return ""


def level_to_channel(ds, short_name, l0=1000):
    if len(ds.level) == 1:
        channel = [short_name]
    else:
        if ds.level.data[0] != l0:
            ds = ds.reindex(level=ds.level[::-1])
        channel = [f'{short_name}{lvl}' for lvl in ds.level.data]
    ds.attrs = {}  
    ds.name = "data"   
    ds = ds.rename({'level': 'channel'})
    ds = ds.assign_coords(channel=channel)  
    return ds

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



def load_cra(file_name, short_name, new_name, levels=[]):

    try:
        ds = pg.open(file_name)

        if len(levels) > 1:
            data = ds.select(shortName=short_name, level=levels)
        else:
            data = ds.select(shortName=short_name)
    except:
        print(f"Load {short_name} failed !")
        return 
    
    for v in data:
        lats = v.distinctLatitudes
        lons = v.distinctLongitudes
        time_str = f'{v.dataDate}'

    # msg = f"time_str: {time_str}"
    # msg += f"lat: {len(lats)}, {lats[0]} ~ {lats[-1]}"
    # msg += f"lon: {len(lons)}, {lons[0]} ~ {lons[-1]}"
    # print(msg)

    imgs = np.zeros((len(levels), len(lats), len(lons)), dtype=np.float32)
    # assert len(data) == len(levels), (len(data))
    
    for v in data:  
        img, _, _ = v.data() 
        level = int(v.level)
        
        if len(levels) == 1:
            imgs[0] = img
        else:
            i = levels.index(level)
            imgs[i] = img

    init_times = [pd.to_datetime(time_str)]
    data = imgs[None] * unit_scale.get(short_name, 1)
    
    v = xr.DataArray(
        name=new_name,
        data=data,
        dims=['time', 'level', 'lat', 'lon'],
        coords={
            'time': init_times, 
            'level': levels, 
            'lat': lats, 
            'lon': lons
        },
    )
    return v


def load_cra40land(file_name, short_name, new_name, levels=[]):

    try:
        data = pg.open(file_name).select()
    except:
        print(f"Load {short_name} failed !")
        return 
    
    # assert len(data) == len(levels), (len(data))
    for v in data:  
        img, lats, lons = v.data() 
        lats = lats[:, 0]
        lons = lons[0, :]
        imgs = np.zeros((len(levels), len(lats), len(lons)), dtype=np.float32)
        imgs[0] = img
        # from IPython import embed; embed()
        time_str = str(v.dataDate)
        init_times = [pd.to_datetime(time_str)]
        new_data = imgs[None] * unit_scale.get(short_name, 1)
        # msg = f"{time_str}, img: {new_data.shape}, {new_data.min():.3f} ~  {new_data.max():.3f}, "
        # msg += f"lat: {len(lats)}, {lats[0]} ~ {lats[-1]}, "
        # msg += f"lon: {len(lons)}, {lons[0]} ~ {lons[-1]}"
        # print(msg)

        v = xr.DataArray(
            name=new_name,
            data=new_data,
            dims=['time', 'level', 'lat', 'lon'],
            coords={
                'time': init_times, 
                'level': levels, 
                'lat': lats, 
                'lon': lons
            },
        )
        return v


def load_other(file_name, short_name, new_name, levels=[]):
    v = xr.open_dataarray(file_name)
    v.name = new_name
    v = v.expand_dims({'level': levels}, axis=0) 
    if new_name == "ttr":
        return -v
    return v


def make_single(data_dir, degree=1.5):
    ds = []
    lat = np.arange(90, -90-degree , -degree)
    lon = np.arange(0, 360, degree)

    for new_name, cra_name in cra_names.items():
        prefix = cra_name['prefix']
        short_name = cra_name['short_name']
        levels = cra_name['levels']
        file_name = get_file_name(data_dir, prefix)

        if new_name == "t2m":
            v = load_cra40land(file_name, short_name, new_name, levels)
        elif new_name in ["sst", "ttr"]:
            v = load_other(file_name, short_name, new_name, levels)
        else:
            v = load_cra(file_name, short_name, new_name, levels)

        v = level_to_channel(v, new_name, l0=1000)
        if v.lat.data[0] < 0:
            v = v.reindex(lat=v.lat[::-1])        
        v = v.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})

        # zero tp
        if new_name == "tp":
            v = v * 0

        ds.append(v)

    ds = xr.concat(ds, 'channel')
    print_dataarray(ds)
    return ds


def make_input(data_dir):
    file_names = []
    for time_str in os.listdir(data_dir):
        time_dir = os.path.join(data_dir, time_str)
        if os.path.isdir(time_dir):
            file_names.append(time_str)
    file_names = sorted(file_names)[-2:]
    print(f"\nMake x1 from {file_names[0]} ...")
    x1 = make_single(file_names[0])
    print(f"\nMake x2 from {file_names[1]} ...")
    x2 = make_single(file_names[1])
    input = xr.concat([x1, x2], "time")
    print_dataarray(input)
    return input

