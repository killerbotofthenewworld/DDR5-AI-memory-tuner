"""
Import configurations from popular overclocking tools
"""
import streamlit as st
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd

class ToolImporter:
    """Handle imports from popular overclocking tools"""
    
    @staticmethod
    def parse_asus_profile(file_content: str) -> Optional[Dict[str, Any]]:
        """Parse ASUS AI Overclock Tuner profile"""
        try:
            # ASUS profiles are typically XML-based
            root = ET.fromstring(file_content)
            
            config = {
                'frequency': 0,
                'cl': 0,
                'trcd': 0,
                'trp': 0,
                'tras': 0,
                'vddq': 1.1,
                'vpp': 1.8,
                'source': 'ASUS AI Overclock Tuner'
            }
            
            # Parse memory settings
            memory_section = root.find('.//Memory')
            if memory_section is not None:
                freq_elem = memory_section.find('Frequency')
                if freq_elem is not None:
                    config['frequency'] = int(freq_elem.text)
                
                timings = memory_section.find('Timings')
                if timings is not None:
                    config['cl'] = int(timings.find('CL').text if timings.find('CL') is not None else 0)
                    config['trcd'] = int(timings.find('tRCD').text if timings.find('tRCD') is not None else 0)
                    config['trp'] = int(timings.find('tRP').text if timings.find('tRP') is not None else 0)
                    config['tras'] = int(timings.find('tRAS').text if timings.find('tRAS') is not None else 0)
                
                voltages = memory_section.find('Voltages')
                if voltages is not None:
                    config['vddq'] = float(voltages.find('VDDQ').text if voltages.find('VDDQ') is not None else 1.1)
                    config['vpp'] = float(voltages.find('VPP').text if voltages.find('VPP') is not None else 1.8)
            
            return config
            
        except Exception as e:
            st.error(f"Error parsing ASUS profile: {e}")
            return None
    
    @staticmethod
    def parse_msi_profile(file_content: str) -> Optional[Dict[str, Any]]:
        """Parse MSI Dragon Center profile"""
        try:
            # MSI profiles are typically JSON-based
            data = json.loads(file_content)
            
            config = {
                'frequency': data.get('memory_frequency', 0),
                'cl': data.get('timings', {}).get('CL', 0),
                'trcd': data.get('timings', {}).get('tRCD', 0),
                'trp': data.get('timings', {}).get('tRP', 0),
                'tras': data.get('timings', {}).get('tRAS', 0),
                'vddq': data.get('voltages', {}).get('VDDQ', 1.1),
                'vpp': data.get('voltages', {}).get('VPP', 1.8),
                'source': 'MSI Dragon Center'
            }
            
            return config
            
        except Exception as e:
            st.error(f"Error parsing MSI profile: {e}")
            return None
    
    @staticmethod
    def parse_gigabyte_profile(file_content: str) -> Optional[Dict[str, Any]]:
        """Parse Gigabyte RGB Fusion profile"""
        try:
            # Gigabyte profiles are typically INI-style
            lines = file_content.strip().split('\n')
            config = {
                'frequency': 0,
                'cl': 0,
                'trcd': 0,
                'trp': 0,
                'tras': 0,
                'vddq': 1.1,
                'vpp': 1.8,
                'source': 'Gigabyte RGB Fusion'
            }
            
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'frequency' in key:
                        config['frequency'] = int(value)
                    elif key == 'cl':
                        config['cl'] = int(value)
                    elif key == 'trcd':
                        config['trcd'] = int(value)
                    elif key == 'trp':
                        config['trp'] = int(value)
                    elif key == 'tras':
                        config['tras'] = int(value)
                    elif 'vddq' in key:
                        config['vddq'] = float(value)
                    elif 'vpp' in key:
                        config['vpp'] = float(value)
            
            return config
            
        except Exception as e:
            st.error(f"Error parsing Gigabyte profile: {e}")
            return None
    
    @staticmethod
    def parse_hwinfo_export(file_content: str) -> Optional[Dict[str, Any]]:
        """Parse HWiNFO64 CSV export"""
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(file_content))
            
            config = {
                'frequency': 0,
                'cl': 0,
                'trcd': 0,
                'trp': 0,
                'tras': 0,
                'vddq': 1.1,
                'vpp': 1.8,
                'source': 'HWiNFO64 Export'
            }
            
            # Look for memory-related entries
            memory_rows = df[df['Sensor Name'].str.contains('Memory|DDR', case=False, na=False)]
            
            for _, row in memory_rows.iterrows():
                label = row['Label'].lower()
                value = row['Value']
                
                if 'frequency' in label or 'speed' in label:
                    config['frequency'] = int(float(value))
                elif 'cl' in label or 'cas' in label:
                    config['cl'] = int(float(value))
                elif 'trcd' in label:
                    config['trcd'] = int(float(value))
                elif 'trp' in label:
                    config['trp'] = int(float(value))
                elif 'tras' in label:
                    config['tras'] = int(float(value))
                elif 'vddq' in label:
                    config['vddq'] = float(value)
                elif 'vpp' in label:
                    config['vpp'] = float(value)
            
            return config
            
        except Exception as e:
            st.error(f"Error parsing HWiNFO export: {e}")
            return None


def create_tool_import_interface():
    """Create the tool import interface"""
    
    st.markdown("### 📥 Import from Popular Tools")
    
    # Tool selection
    supported_tools = {
        'ASUS AI Overclock Tuner': 'asus',
        'MSI Dragon Center': 'msi', 
        'Gigabyte RGB Fusion': 'gigabyte',
        'HWiNFO64 Export': 'hwinfo',
        'CPU-Z Validation': 'cpuz',
        'AIDA64 Report': 'aida64'
    }
    
    selected_tool = st.selectbox(
        "Select Tool:",
        list(supported_tools.keys()),
        format_func=lambda x: f"🔧 {x}"
    )
    
    tool_code = supported_tools[selected_tool]
    
    # File upload
    uploaded_file = st.file_uploader(
        f"Upload {selected_tool} profile/export file",
        type=['xml', 'json', 'txt', 'csv', 'ini'],
        help=f"Upload a configuration file from {selected_tool}"
    )
    
    if uploaded_file is not None:
        try:
            # Read file content
            file_content = uploaded_file.read().decode('utf-8')
            
            # Parse based on tool type
            importer = ToolImporter()
            config = None
            
            if tool_code == 'asus':
                config = importer.parse_asus_profile(file_content)
            elif tool_code == 'msi':
                config = importer.parse_msi_profile(file_content)
            elif tool_code == 'gigabyte':
                config = importer.parse_gigabyte_profile(file_content)
            elif tool_code == 'hwinfo':
                config = importer.parse_hwinfo_export(file_content)
            else:
                st.warning(f"Parser for {selected_tool} not yet implemented")
                return
            
            if config:
                st.success(f"✅ Successfully imported from {selected_tool}!")
                
                # Display imported configuration
                st.markdown("#### 📋 Imported Configuration:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Frequency", f"{config['frequency']} MT/s")
                    st.metric("CL", config['cl'])
                    st.metric("tRCD", config['trcd'])
                
                with col2:
                    st.metric("tRP", config['trp'])
                    st.metric("tRAS", config['tras'])
                    st.metric("VDDQ", f"{config['vddq']:.3f}V")
                
                with col3:
                    st.metric("VPP", f"{config['vpp']:.3f}V")
                    st.info(f"Source: {config['source']}")
                
                # Apply to current session
                if st.button("Apply to Current Session", type="primary"):
                    st.session_state['imported_config'] = config
                    st.success("Configuration applied to current session!")
                    st.rerun()
                
                # Save as preset
                if st.button("Save as Preset"):
                    preset_name = st.text_input("Preset Name:", f"Import from {selected_tool}")
                    if preset_name:
                        if 'saved_presets' not in st.session_state:
                            st.session_state['saved_presets'] = {}
                        st.session_state['saved_presets'][preset_name] = config
                        st.success(f"Saved as preset: {preset_name}")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Import instructions
    with st.expander("📖 Import Instructions"):
        st.markdown(f"""
        ### How to export from {selected_tool}:
        
        **ASUS AI Overclock Tuner:**
        - Open ASUS AI Suite
        - Go to AI Overclock Tuner
        - Click "Export Profile" or "Save Settings"
        - Save as XML file
        
        **MSI Dragon Center:**
        - Open MSI Dragon Center
        - Go to Mystic Light or Overclocking
        - Export profile as JSON
        
        **Gigabyte RGB Fusion:**
        - Open RGB Fusion software
        - Go to Memory settings
        - Export configuration
        
        **HWiNFO64:**
        - Run HWiNFO64
        - Go to Sensors window
        - Right-click and select "Save Report"
        - Save as CSV format
        
        **Supported File Formats:**
        - XML (ASUS, generic)
        - JSON (MSI, generic)
        - CSV (HWiNFO, monitoring tools)
        - INI/TXT (Gigabyte, generic)
        """)


def create_export_interface():
    """Create configuration export interface"""
    
    st.markdown("### 📤 Export Current Configuration")
    
    # Get current config from session state
    current_config = st.session_state.get('current_config', {
        'frequency': 5600,
        'cl': 32,
        'trcd': 32,
        'trp': 32,
        'tras': 64,
        'vddq': 1.2,
        'vpp': 1.85
    })
    
    export_format = st.selectbox(
        "Export Format:",
        ['JSON', 'XML', 'CSV', 'TXT'],
        format_func=lambda x: f"📄 {x} Format"
    )
    
    if st.button("Generate Export File"):
        if export_format == 'JSON':
            export_data = json.dumps(current_config, indent=2)
            filename = "ddr5_config.json"
            
        elif export_format == 'XML':
            export_data = f"""<?xml version="1.0" encoding="UTF-8"?>
<DDR5Configuration>
    <Memory>
        <Frequency>{current_config['frequency']}</Frequency>
        <Timings>
            <CL>{current_config['cl']}</CL>
            <tRCD>{current_config['trcd']}</tRCD>
            <tRP>{current_config['trp']}</tRP>
            <tRAS>{current_config['tras']}</tRAS>
        </Timings>
        <Voltages>
            <VDDQ>{current_config['vddq']}</VDDQ>
            <VPP>{current_config['vpp']}</VPP>
        </Voltages>
    </Memory>
</DDR5Configuration>"""
            filename = "ddr5_config.xml"
            
        elif export_format == 'CSV':
            export_data = "Parameter,Value\n"
            for key, value in current_config.items():
                export_data += f"{key},{value}\n"
            filename = "ddr5_config.csv"
            
        else:  # TXT
            export_data = "DDR5 Configuration Export\n"
            export_data += "=" * 30 + "\n\n"
            for key, value in current_config.items():
                export_data += f"{key.upper()}: {value}\n"
            filename = "ddr5_config.txt"
        
        # Provide download button
        st.download_button(
            label=f"📥 Download {filename}",
            data=export_data,
            file_name=filename,
            mime='text/plain'
        )
        
        st.success(f"Export file generated successfully!")


def create_tool_imports_interface():
    """Create the main tool imports interface - wrapper function"""
    st.subheader("🔄 Import/Export Popular Tools")
    
    tab1, tab2 = st.tabs(["📥 Import", "📤 Export"])
    
    with tab1:
        create_tool_import_interface()
    
    with tab2:
        create_export_interface()


if __name__ == "__main__":
    st.set_page_config(page_title="Tool Import Test", layout="wide")
    st.title("🔧 Tool Import/Export Test")
    
    create_tool_imports_interface()
