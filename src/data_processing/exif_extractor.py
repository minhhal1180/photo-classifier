import os
import csv
import subprocess
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

class ExifExtractor:
    def __init__(self, exiftool_path: Optional[str] = None):
        self.exiftool_path = exiftool_path or 'exiftool'
        
        # Danh sách fields cần trích xuất
        self.selected_fields = [
            # Basic File Info
            'FileName',
            'FileType',
            
            # Camera & Lens Info
            'Make',
            'Model',
            'LensModel',
            'Orientation',
            
            # Dates
            'CreateDate',
            
            # Exposure & Light
            'ShutterSpeed',
            'FNumber', 
            'ISO',
            'LightValue',
            'Flash',
            
            # Distances & Optics
            'FocalLength',
            'FocalLengthIn35mmFormat',
            'FocusDistance',
            'FocusDistance2',
            'FocusDistanceLower',
            'FocusDistanceUpper',
            'HyperfocalDistance',
            'CircleOfConfusion',
            'DepthOfField',
        ]
        
        self.image_extensions = (
            '.arw', '.cr3', '.cr2', '.jpg', '.jpeg', '.png', 
            '.tiff', '.tif', '.nef', '.dng', '.raw', '.rw2', 
            '.raf', '.orf'
        )
    
    def extract_from_folder(
        self, 
        input_folder: Path, 
        output_folder: Path,
        recursive: bool = True
    ) -> Optional[Path]:
        output_folder.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_csv = output_folder / f'temp_raw_{timestamp}.csv'
        final_output_file = output_folder / f'exif_data_complete_{timestamp}.csv'
        
        try:
            # Build ExifTool command
            cmd = [self.exiftool_path, '-csv', '-n']
            if recursive:
                cmd.append('-r')
            
            cmd.extend([f'-{f}' for f in self.selected_fields])
            cmd.append(str(input_folder))
            
            # Run ExifTool
            with open(temp_csv, 'w', encoding='utf-8') as outfile:
                result = subprocess.run(
                    cmd, 
                    stdout=outfile, 
                    stderr=subprocess.PIPE, 
                    text=True
                )
            
            if result.returncode not in (0, 1):
                raise subprocess.CalledProcessError(
                    result.returncode, 
                    cmd, 
                    stderr=result.stderr
                )
            
            if result.stderr:
                print(f"ExifTool warnings: {result.stderr.strip()}")
            
            # Post-process data
            self._post_process_csv(temp_csv, final_output_file)
            
            # Clean up temp file
            temp_csv.unlink()
            
            print(f"EXIF extracted: {final_output_file}")
            return final_output_file
            
        except subprocess.CalledProcessError as e:
            error_details = e.stderr.strip() if e.stderr else str(e)
            print(f"ExifTool Error: {error_details}")
            return None
            
        except FileNotFoundError:
            print("Error: ExifTool not found. Please install ExifTool.")
            return None
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def _post_process_csv(self, input_csv: Path, output_csv: Path):
        with open(input_csv, 'r', encoding='utf-8') as f_in, \
             open(output_csv, 'w', newline='', encoding='utf-8-sig') as f_out:
            
            reader = csv.DictReader(f_in)
            
            # Build output fieldnames
            raw_fieldnames = reader.fieldnames or []
            excluded_fields = {
                'FocalLengthIn35mmFormat',
                'FocusDistance2',
                'FocusDistanceLower',
                'FocusDistanceUpper',
                'DepthOfField',
            }
            base_fieldnames = [f for f in raw_fieldnames if f not in excluded_fields]
            
            extra_columns = [
                'OrientationDescription',
                'FlashDescription',
                'Consolidated_Focus_Dist_m',
                'DepthOfField',
                'Depth Of Field',
            ]
            for extra in extra_columns:
                if extra not in base_fieldnames:
                    base_fieldnames.append(extra)
            
            writer = csv.DictWriter(f_out, fieldnames=base_fieldnames)
            writer.writeheader()
            
            for row in reader:
                self._process_row(row)
                writer.writerow(row)
    
    def _process_row(self, row: Dict[str, str]):
        # A. Translate orientation and flash codes
        row['OrientationDescription'] = self._orientation_description(
            row.get('Orientation')
        )
        row['FlashDescription'] = self._describe_flash(row.get('Flash'))
        
        # B. Merge focal length variants
        focal_length_val = self._parse_float(row.get('FocalLength'))
        focal_length_35_val = self._parse_float(row.get('FocalLengthIn35mmFormat'))
        
        if focal_length_35_val and (not focal_length_val or focal_length_35_val > focal_length_val):
            row['FocalLength'] = self._format_number(focal_length_35_val)
        elif focal_length_val:
            row['FocalLength'] = self._format_number(focal_length_val)
        else:
            row['FocalLength'] = ''
        
        # C. Consolidate focus distance
        focus_distance = row.get('FocusDistance')
        
        if not focus_distance:
            if row.get('FocusDistance2'):
                focus_distance = row.get('FocusDistance2')
            else:
                lower_val = self._parse_float(row.get('FocusDistanceLower'))
                upper_val = self._parse_float(row.get('FocusDistanceUpper'))
                if lower_val and upper_val:
                    focus_distance = (lower_val + upper_val) / 2
                elif lower_val:
                    focus_distance = lower_val
                elif upper_val:
                    focus_distance = upper_val
        
        focus_distance_str = '-1'
        if focus_distance:
            focus_distance_value = self._parse_float(focus_distance)
            if focus_distance_value is not None:
                focus_distance_str = self._format_number(focus_distance_value, missing_marker='-1')
        
        row['FocusDistance'] = focus_distance_str
        row['Consolidated_Focus_Dist_m'] = focus_distance_str
        
        # D. Calculate depth of field
        dof_value = row.get('DepthOfField')
        focus_distance_numeric = self._parse_float(row.get('FocusDistance'))
        valid_focus_distance = (
            focus_distance_numeric is not None and 
            not math.isnan(focus_distance_numeric) and 
            focus_distance_numeric > 0
        )
        
        if row['FocalLength'] and row.get('FNumber') and valid_focus_distance:
            coc = row.get('CircleOfConfusion') or 0.029
            calculated = self._calculate_dof(
                row['FocalLength'], 
                row.get('FNumber'), 
                focus_distance_numeric, 
                coc
            )
            if calculated is not None:
                dof_value = calculated
                row['DepthOfField'] = calculated
        
        if dof_value:
            formatted_dof = self._format_number(dof_value)
            row['Depth Of Field'] = formatted_dof
            row['DepthOfField'] = formatted_dof
        else:
            row['Depth Of Field'] = ''
            row['DepthOfField'] = ''
        
        # E. Remove helper fields
        row.pop('FocalLengthIn35mmFormat', None)
        row.pop('FocusDistance2', None)
        row.pop('FocusDistanceLower', None)
        row.pop('FocusDistanceUpper', None)
    
    @staticmethod
    def _parse_float(value) -> Optional[float]:
        if value is None:
            return None
        value = str(value).strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    
    @staticmethod
    def _format_number(value, missing_marker=None) -> str:
        if value in (None, ''):
            return missing_marker if missing_marker is not None else ''
        
        try:
            num = float(value)
        except (TypeError, ValueError):
            return str(value)
        
        if math.isinf(num):
            num = 1000.0
        if math.isnan(num):
            return missing_marker if missing_marker is not None else ''
        
        formatted = f"{num:.3f}".rstrip('0').rstrip('.')
        return formatted if formatted else '0'
    
    @staticmethod
    def _orientation_description(value) -> Optional[str]:
        orientation_map = {
            1: "Horizontal (normal)",
            2: "Mirror horizontal",
            3: "Rotate 180",
            4: "Mirror vertical",
            5: "Mirror horizontal and rotate 270 CW",
            6: "Rotate 90 CW",
            7: "Mirror horizontal and rotate 90 CW",
            8: "Rotate 270 CW",
        }
        code = ExifExtractor._parse_float(value)
        if code is None:
            return None
        return orientation_map.get(int(code), "Unknown orientation")
    
    @staticmethod
    def _describe_flash(value) -> Optional[str]:
        code = ExifExtractor._parse_float(value)
        if code is None:
            return None
        
        code_int = int(code)
        if code_int in (0, 1):
            return "Flash fired" if code_int == 1 else "Flash did not fire"
        
        fired = bool(code_int & 0x1)
        parts = ["Flash fired" if fired else "Flash did not fire"]
        
        if code_int & 0x4:
            parts.append("return not detected")
        elif code_int & 0x6 == 0x6:
            parts.append("return detected")
        
        if code_int & 0x8:
            parts.append("flash on")
        if code_int & 0x10:
            parts.append("compulsory mode")
        if code_int & 0x20:
            parts.append("red-eye reduction")
        if code_int & 0x40:
            parts.append("auto mode")
        
        return ", ".join(parts)
    
    @staticmethod
    def _calculate_dof(focal_length_mm, f_number, subject_distance_m, coc_mm=0.029) -> Optional[float]:
        try:
            if any(v is None or float(v) == 0 for v in [focal_length_mm, f_number, subject_distance_m]):
                return None
            
            fl = float(focal_length_mm)
            fn = float(f_number)
            dist = float(subject_distance_m)
            coc = float(coc_mm)
            
            # Hyperfocal Distance
            H = (fl ** 2) / (fn * coc) + fl
            H_m = H / 1000.0
            
            # Near Limit
            dn = (H_m * dist) / (H_m + (dist - (fl/1000.0)))
            
            # Far Limit
            if (dist - (fl/1000.0)) >= H_m:
                df = float('inf')
                dof = float('inf')
            else:
                df = (H_m * dist) / (H_m - (dist - (fl/1000.0)))
                dof = df - dn
            
            return round(dof, 3)
        except Exception:
            return None

def extract_exif_from_images(
    input_folder: str | Path,
    output_folder: str | Path,
    exiftool_path: Optional[str] = None
) -> Optional[pd.DataFrame]:
    extractor = ExifExtractor(exiftool_path=exiftool_path)
    csv_path = extractor.extract_from_folder(
        Path(input_folder), 
        Path(output_folder)
    )
    
    if csv_path and csv_path.exists():
        return pd.read_csv(csv_path)
    
    return None
