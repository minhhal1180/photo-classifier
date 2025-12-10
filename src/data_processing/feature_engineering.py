import numpy as np
import pandas as pd
from typing import List, Optional

class FeatureEngineer:
    def __init__(self, feature_columns: Optional[List[str]] = None):
        if feature_columns is None:
            self.feature_columns = [
                'FocalLength_35mm',
                'FocusDist_Clean',
                'DoF_Clean',
                'HyperfocalRatio',
                'FNumber',
                'LogShutterSpeed',
                'ISO',
                'LightValue',
                'Hour',
                'Flash_Binary',
            ]
        else:
            self.feature_columns = feature_columns
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Focus Distance
        df = self._process_focus_distance(df)
        
        # 2. Depth of Field
        df = self._process_dof(df)
        
        # 3. Hyperfocal Ratio
        df = self._process_hyperfocal_ratio(df)
        
        # 4. Shutter Speed (log transform)
        df = self._process_shutter_speed(df)
        
        # 5. ISO
        df = self._process_iso(df)
        
        # 6. Light Value
        df = self._process_light_value(df)
        
        # 7. Focal Length (35mm equivalent)
        df = self._process_focal_length(df)
        
        # 8. Hour (from CreateDate)
        df = self._process_hour(df)
        
        # 9. Flash Binary
        df = self._process_flash(df)
        
        # 10. F-Number (already clean, just handle missing)
        if 'FNumber' not in df.columns:
            df['FNumber'] = np.nan
        df['FNumber'] = pd.to_numeric(df['FNumber'], errors='coerce').fillna(5.6)
        
        # Ensure all feature columns exist
        for column in self.feature_columns:
            if column not in df.columns:
                df[column] = np.nan
        
        return df
    
    def _process_focus_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        focus_col = None
        if 'Consolidated_Focus_Dist_m' in df.columns:
            focus_col = 'Consolidated_Focus_Dist_m'
        elif 'FocusDistance' in df.columns:
            focus_col = 'FocusDistance'
        
        if focus_col:
            df['FocusDist_Clean'] = pd.to_numeric(df[focus_col], errors='coerce')
        else:
            df['FocusDist_Clean'] = np.nan
        
        # Replace invalid values with 1000 (infinity)
        df['FocusDist_Clean'] = df['FocusDist_Clean'].fillna(1000)
        df.loc[df['FocusDist_Clean'] < 0, 'FocusDist_Clean'] = 1000
        
        return df
    
    def _process_dof(self, df: pd.DataFrame) -> pd.DataFrame:
        dof_series = None
        for candidate in ('DepthOfField', 'Depth Of Field'):
            if candidate in df.columns:
                dof_series = pd.to_numeric(df[candidate], errors='coerce')
                break
        
        if dof_series is None:
            dof_series = pd.Series(np.nan, index=df.index)
        
        df['DoF_Clean'] = dof_series.fillna(-1)
        
        return df
    
    def _process_hyperfocal_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'HyperfocalDistance' in df.columns:
            df['HyperfocalDistance'] = pd.to_numeric(
                df['HyperfocalDistance'], errors='coerce'
            )
        else:
            df['HyperfocalDistance'] = np.nan
        
        df['HyperfocalDistance'] = df['HyperfocalDistance'].fillna(1000)
        
        # Avoid division by zero
        df['HyperfocalRatio'] = df['FocusDist_Clean'] / df['HyperfocalDistance']
        df['HyperfocalRatio'] = df['HyperfocalRatio'].replace([np.inf, -np.inf], 1.0)
        
        return df
    
    def _process_shutter_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'ShutterSpeed' in df.columns:
            shutter = pd.to_numeric(df['ShutterSpeed'], errors='coerce')
            shutter = shutter.replace(0, 0.0001)
        else:
            shutter = pd.Series(0.0001, index=df.index)
        
        df['LogShutterSpeed'] = np.log(shutter)
        
        return df
    
    def _process_iso(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'ISO' in df.columns:
            df['ISO'] = pd.to_numeric(df['ISO'], errors='coerce')
        else:
            df['ISO'] = np.nan
        
        df['ISO'] = df['ISO'].fillna(100)
        
        return df
    
    def _process_light_value(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'LightValue' in df.columns:
            df['LightValue'] = pd.to_numeric(
                df['LightValue'], errors='coerce'
            ).fillna(0)
        else:
            df['LightValue'] = 0
        
        return df
    
    def _process_focal_length(self, df: pd.DataFrame) -> pd.DataFrame:
        df['FocalLength_35mm'] = pd.to_numeric(
            df.get('FocalLength', np.nan), errors='coerce'
        )
        
        return df
    
    def _process_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        def get_hour(value: str) -> int:
            try:
                return int(str(value).split(' ')[1].split(':')[0])
            except Exception:
                return -1
        
        if 'CreateDate' in df.columns:
            df['Hour'] = df['CreateDate'].apply(get_hour)
        else:
            df['Hour'] = -1
        
        return df
    
    def _process_flash(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'FlashDescription' in df.columns:
            flash_series = df['FlashDescription']
        elif 'Flash' in df.columns:
            flash_series = df['Flash']
        else:
            flash_series = pd.Series('', index=df.index)
        
        df['Flash_Binary'] = flash_series.apply(
            lambda x: 1 if 'fired' in str(x).lower() and 
                          'did not fire' not in str(x).lower() 
                      else 0
        )
        
        return df

def engineer_features(
    df: pd.DataFrame, 
    feature_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    engineer = FeatureEngineer(feature_columns=feature_columns)
    return engineer.engineer_features(df)
