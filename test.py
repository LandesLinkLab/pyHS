import numpy as np
from pathlib import Path
from nptdms import TdmsFile
import os

def inspect_tdms_file(file_path):
    """TDMS 파일 구조 상세 분석"""
    print(f"\n=== Inspecting TDMS file: {Path(file_path).name} ===\n")
    
    td = TdmsFile.read(file_path)
    
    # 1. Root properties
    print("1. ROOT PROPERTIES:")
    for prop, value in td.properties.items():
        print(f"   {prop}: {value}")
    
    # 2. Groups and channels
    print("\n2. STRUCTURE:")
    for group in td.groups():
        print(f"\nGroup: '{group.name}'")
        channels = list(group.channels())
        print(f"  Number of channels: {len(channels)}")
        
        for i, ch in enumerate(channels[:5]):
            data = ch[:]
            print(f"  Channel {i}: shape={data.shape}, dtype={data.dtype}")
            if data.ndim == 1 and len(data) < 10:
                print(f"    Values: {data}")
            elif data.ndim == 1:
                print(f"    Range: [{data.min():.3f}, {data.max():.3f}], Length: {len(data)}")
                print(f"    First 5: {data[:5]}")
                print(f"    Last 5: {data[-5:]}")
        
        if len(channels) > 5:
            print(f"  ... and {len(channels)-5} more channels")
    
    # 3. Look for wavelength info
    print("\n3. SEARCHING FOR WAVELENGTH INFORMATION:")
    
    if 'Info' in td:
        info_group = td['Info']
        print("\n  In 'Info' group:")
        for ch in info_group.channels():
            ch_name = ch.name
            data = ch[:]
            print(f"    Channel '{ch_name}': shape={data.shape}")
            if data.ndim == 1 and len(data) < 2000:
                print(f"      Range: [{data.min():.3f}, {data.max():.3f}]")
                if 'wave' in ch_name.lower() or 'wl' in ch_name.lower() or 'nm' in ch_name.lower():
                    print(f"      >>> POTENTIAL WAVELENGTH CHANNEL <<<")
    
    print("\n  Checking all 1D arrays:")
    all_channels = [ch for g in td.groups() for ch in g.channels()]
    for ch in all_channels:
        data = ch[:]
        if data.ndim == 1 and 100 < len(data) < 2000:
            if 400 < data.min() < 600 and 800 < data.max() < 1200:
                print(f"    Found potential wavelength array in '{ch.name}':")
                print(f"      Length: {len(data)}, Range: [{data.min():.1f}, {data.max():.1f}] nm")
                print(f"      Monotonic: {np.all(np.diff(data) > 0)}")
    
    # 4. Check metadata for wavelength calculation
    print("\n4. WAVELENGTH FROM METADATA:")
    if 'center wvlth wavelength' in td.properties:
        center = td.properties['center wvlth wavelength']
        print(f"   Center wavelength: {center} nm")
    if 'wvlth group' in td.properties:
        wvlth_group = td.properties['wvlth group']
        print(f"   Wavelength group: {wvlth_group}")
    
    # Common hyperspectral calculation
    if 'Spectra' in td:
        spectra_group = td['Spectra']
        spectra_channels = list(spectra_group.channels())
        if spectra_channels:
            ch = spectra_channels[0]
            spectrum_length = len(ch[:])
            print(f"   Spectrum length: {spectrum_length} points")
            
            if spectrum_length == 670:
                print("\n   >>> CALCULATING WAVELENGTH ARRAY <<<")
                print(f"   Assuming 550–1000 nm range with {spectrum_length} points:")
                wl_calculated = np.linspace(550, 1000, spectrum_length)
                print(f"   First 5: {wl_calculated[:5]}")
                print(f"   Last 5: {wl_calculated[-5:]}")
    

# Main execution
if __name__ == "__main__":
    home = str(Path.home())
    data_dir = os.path.join(home, 'dataset/pyHS/raw_test')
    
    files = ['AuNR_PMMA.tdms', 'wc.tdms', 'dc.tdms']
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            inspect_tdms_file(file_path)
            print("\n" + "="*80)

