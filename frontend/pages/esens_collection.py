import streamlit as st # type: ignore
import streamlit.components.v1 as components # type: ignore


################## ESENS COLLECTION PAGE ###################
# 1. Esens Collection Page: Main function, sets up title and renders javascript data collection interface using 
#  Streamlit's components.html()
# 2. Render Data Collection HTML: Helper function returning HTML and JavaScript code for 
# the eSense data collection interface. Creates a self-contained web application that runs within the Streamlit page
# Includes--> Connect, Start Sampling, Stop Sampling, and Download Data buttons, status indicator and data display

# Java Script Function: 
# 1. Calculate Checksum: Calculates a 8-bit checksum for commands sent to the device, then sums all values and 
#    applies a bitwise AND with 0xFF to get only the least significant byte
# 2. Start Sampling:  Createa a binary command to start sampling data at 50Hz, returning a Uint8Array 
# 3. Stop Sampling: Binary command to stop sampling data using command code 0x00 
# 4. Parse IMU Data: Defines scaling factors and uses to convert accelerometer/gyroscope readings. Also saves
#    readable timestamps
# 5. Download Data: Checks if data has been collecte and creates a timestamp for the filename. 
def esens_collection_page():
    # Render the HTML for the eSense data collection
    st.header("eSense Data Collection")
    st.write("Use the web application to record data from your eSense earbuds.")
    
    # Import and use the render_data_collection_html function from your paste.txt
    components.html(render_data_collection_html(), height=600)


def render_data_collection_html():
    return """
    <div style="padding: 20px;">
        <style>
            .esense-container button { 
                padding: 10px 20px; 
                margin: 5px;
                font-size: 16px;
                cursor: pointer;
            }
            .esense-container button:disabled { cursor: not-allowed; }
            #dataDisplay { 
                font-family: monospace;
                white-space: pre;
                margin-top: 20px;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
            .status {
                margin-top: 10px;
                color: #666;
            }
            #recordingStatus {
                color: #ff0000;
                font-weight: bold;
                display: none;
            }
        </style>
        
        <div class="esense-container">
            <button id="connectButton">Connect to eSense</button>
            <button id="startButton" disabled>Start Sampling</button>
            <button id="stopButton" disabled>Stop Sampling</button>
            <button id="downloadButton" disabled>Download Data</button>
            
            <div class="status">
                <span id="recordingStatus">● Recording</span>
                <div id="sampleCount">Samples collected: 0</div>
            </div>

            <div id="dataDisplay"></div>
        </div>

       
        <script>
            // Global variables for storing device connection and data

            let device;                // Bluetooth device
            let imuDataCharacteristic; // IMU data specififcally
            let configCharacteristic;  // Characteristic for configuration
            let recorded_data = [];     // Array to store IMU data
            let isRecording = false;   // Flag track ingrecording state
            let recordingStartTime = null; // Timestamp when recording started
            
            function calculateChecksum(dataSize, ...data) {
                const sum = dataSize + data.reduce((a, b) => a + b, 0);
                return sum & 0xFF; // Bitwise AND to get last 8 bits
            }

            function startSamplingCommand(rate = 50) {
                return new Uint8Array([
                    0x53,                           // Command header
                    calculateChecksum(0x02, 0x01, rate), // Checksum
                    0x02,                           // Data size
                    0x01,                           // Start sampling command code
                    rate                            // Sampling rate
                ]);
            }

            function stopSamplingCommand() {
                return new Uint8Array([
                    0x53,                           // Command header
                    calculateChecksum(0x02, 0x00, 0x00), // Checksum
                    0x02,                           // Data size
                    0x00,                           // Stop sampling command code
                    0x00                            // Padding
                ]);
            }

            
            // Parse raw IMU data from the device
            function parseIMUData(data) {
                function bytesToInt16(high, low) {
                    const value = (high << 8) | low;
                    return value > 0x7FFF ? value - 0x10000 : value;
                }

                // Scaling factors for converting raw values to physical units
                const accelerometer_scaling = 8192.0;
                const gyroscope_scaling = 65.5;

                // Create timestamp for the data point
                const timestamp = Date.now();
                const date = new Date(timestamp);
                const formattedTime = [
                    date.getHours().toString().padStart(2, '0'),
                    date.getMinutes().toString().padStart(2, '0'),
                    date.getSeconds().toString().padStart(2, '0'),
                    date.getMilliseconds().toString().padStart(3, '0')
                ].join(':');

                 // Calculate time since recording started
                const seconds_since_recording_time = recordingStartTime ? (timestamp - recordingStartTime) / 1000 : 0;

                // parse and return IMU data in correct structure

                return {
                    timestamp: formattedTime,
                    packetIndex: data.getUint8(1),
                    gyro: {
                        x: bytesToInt16(data.getUint8(3), data.getUint8(4)) / gyroscope_scaling,
                        y: bytesToInt16(data.getUint8(5), data.getUint8(6)) / gyroscope_scaling,
                        z: bytesToInt16(data.getUint8(7), data.getUint8(8)) / gyroscope_scaling
                    },
                    accel: {
                        x: bytesToInt16(data.getUint8(9), data.getUint8(10)) / accelerometer_scaling,
                        y: bytesToInt16(data.getUint8(11), data.getUint8(12)) / accelerometer_scaling,
                        z: bytesToInt16(data.getUint8(13), data.getUint8(14)) / accelerometer_scaling
                    }
                };
            }
             // download recorded IMU data as a CSV file
            function download_data() {
                if (recorded_data.length === 0) {
                    alert('No data to download');
                    return;
                }

                const timestamp = new Date().toISOString() // use timestamp in the filename
                    .replace(/[:.]/g, '')
                    .slice(0, -4);

                // start building csv with headers 
                const rows = ["timestamp,x-axis (g),y-axis (g),z-axis (g),x-axis (deg/s),y-axis (deg/s),z-axis (deg/s)"];
                recorded_data.forEach(data => {
                    rows.push(`${data.timestamp},${data.accel.x.toFixed(6)},${data.accel.y.toFixed(6)},${data.accel.z.toFixed(6)},${data.gyro.x.toFixed(6)},${data.gyro.y.toFixed(6)},${data.gyro.z.toFixed(6)}`);
                });

                // use a blob to download data 

                const blob = new Blob([rows.join('\\n')], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.setAttribute('href', url);
                a.setAttribute('download', `esense_imu_data_${timestamp}.csv`);
                a.click();
                window.URL.revokeObjectURL(url);
            }

            // connect to ear buds using web api
            async function connectToESense() {
                try {
                    device = await navigator.bluetooth.requestDevice({ // Request Bluetooth device with eSense filter
                        filters: [{ namePrefix: 'eSense' }],
                        optionalServices: ['0000ff06-0000-1000-8000-00805f9b34fb']
                    });

                    const server = await device.gatt.connect();
                    const service = await server.getPrimaryService('0000ff06-0000-1000-8000-00805f9b34fb');
                    
                     // Get required data for configuration

                    configCharacteristic = await service.getCharacteristic('0000ff07-0000-1000-8000-00805f9b34fb');
                    imuDataCharacteristic = await service.getCharacteristic('0000ff08-0000-1000-8000-00805f9b34fb');

                    // Update UI to reflect connected state
                    document.getElementById('startButton').disabled = false;
                    document.getElementById('connectButton').disabled = true;
                    console.log('Connected to eSense!');
                } catch (error) {
                    console.error('Error connecting:', error);
                    alert('Failed to connect: ' + error.message);
                }
            }

            async function startSampling() {
                try {
                    recorded_data = [];
                    isRecording = true;
                    recordingStartTime = Date.now();
                    
                    await imuDataCharacteristic.startNotifications();
                    imuDataCharacteristic.addEventListener('characteristicvaluechanged', handleIMUData);
                    
                    await configCharacteristic.writeValue(startSamplingCommand(50));
                    
                    document.getElementById('startButton').disabled = true;
                    document.getElementById('stopButton').disabled = false;
                    document.getElementById('downloadButton').disabled = true;
                    document.getElementById('recordingStatus').style.display = 'inline';
                    
                } catch (error) {
                    console.error('Error starting sampling:', error);
                    alert('Failed to start sampling: ' + error.message);
                }
            }

            async function stopSampling() {
                try {
                    isRecording = false;
                    await configCharacteristic.writeValue(stopSamplingCommand());
                    await imuDataCharacteristic.stopNotifications();
                    
                    document.getElementById('startButton').disabled = false;
                    document.getElementById('stopButton').disabled = true;
                    document.getElementById('downloadButton').disabled = false;
                    document.getElementById('recordingStatus').style.display = 'none';
                    
                } catch (error) {
                    console.error('Error stopping sampling:', error);
                    alert('Failed to stop sampling: ' + error.message);
                }
            }

            function handleIMUData(event) {
                const data = parseIMUData(event.target.value);
                
                if (isRecording) {
                    recorded_data.push(data);
                    document.getElementById('sampleCount').textContent = 
                        `Samples collected: ${recorded_data.length}`;
                }

                document.getElementById('dataDisplay').textContent = 
                    `Latest IMU Data:\\n` +
                    `Time: ${data.timestamp}\\n` +
                    `Accelerometer (g): x=${data.accel.x.toFixed(3)}, y=${data.accel.y.toFixed(3)}, z=${data.accel.z.toFixed(3)}\\n` +
                    `Gyroscope (deg/s): x=${data.gyro.x.toFixed(3)}, y=${data.gyro.y.toFixed(3)}, z=${data.gyro.z.toFixed(3)}`;
            }

            // Add button event listeners for UI buttons connecting to corresponding function
            document.getElementById('connectButton').addEventListener('click', connectToESense);
            document.getElementById('startButton').addEventListener('click', startSampling);
            document.getElementById('stopButton').addEventListener('click', stopSampling);
            document.getElementById('downloadButton').addEventListener('click', download_data);
        </script>
    </div>
    """

if __name__ == "__main__":
    esens_collection_page()