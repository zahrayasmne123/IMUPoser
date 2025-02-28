import streamlit as st # type: ignore
import streamlit.components.v1 as components # type: ignore

def esens_collection_page():
    """Display the eSense data collection page."""
    
    # Render the HTML for the eSense data collection
    st.header("eSense Data Collection")
    st.write("Use the web application to record data from your eSense earbuds.")
    
    # Import and use the render_data_collection_html function from your paste.txt
    components.html(render_data_collection_html(), height=600)

    st.markdown("*Finished collecting data? Navigate back to sensor configuration.*")
    if st.button("← Back to Sensor Device Setup"):
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'main'
        
        st.session_state.current_page = 'main'
        st.experimental_rerun()

def render_data_collection_html():
    """Render the HTML for eSense data collection."""
    # Use the HTML content from your paste.txt
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
            let device;
            let imuDataCharacteristic;
            let configCharacteristic;
            let recordedData = [];
            let isRecording = false;
            let recordingStartTime = null;
            
            function calculateChecksum(dataSize, ...data) {
                const sum = dataSize + data.reduce((a, b) => a + b, 0);
                return sum & 0xFF;
            }

            function startSamplingCommand(rate = 50) {
                return new Uint8Array([
                    0x53,
                    calculateChecksum(0x02, 0x01, rate),
                    0x02,
                    0x01,
                    rate
                ]);
            }

            function stopSamplingCommand() {
                return new Uint8Array([
                    0x53,
                    calculateChecksum(0x02, 0x00, 0x00),
                    0x02,
                    0x00,
                    0x00
                ]);
            }

            function parseIMUData(data) {
                function bytesToInt16(high, low) {
                    const value = (high << 8) | low;
                    return value > 0x7FFF ? value - 0x10000 : value;
                }

                const ACCEL_SCALE = 8192.0;
                const GYRO_SCALE = 65.5;

                const timestamp = Date.now();
                const date = new Date(timestamp);
                const formattedTime = [
                    date.getHours().toString().padStart(2, '0'),
                    date.getMinutes().toString().padStart(2, '0'),
                    date.getSeconds().toString().padStart(2, '0'),
                    date.getMilliseconds().toString().padStart(3, '0')
                ].join(':');

                const elapsedSeconds = recordingStartTime ? (timestamp - recordingStartTime) / 1000 : 0;

                return {
                    timestamp: formattedTime,
                    packetIndex: data.getUint8(1),
                    gyro: {
                        x: bytesToInt16(data.getUint8(3), data.getUint8(4)) / GYRO_SCALE,
                        y: bytesToInt16(data.getUint8(5), data.getUint8(6)) / GYRO_SCALE,
                        z: bytesToInt16(data.getUint8(7), data.getUint8(8)) / GYRO_SCALE
                    },
                    accel: {
                        x: bytesToInt16(data.getUint8(9), data.getUint8(10)) / ACCEL_SCALE,
                        y: bytesToInt16(data.getUint8(11), data.getUint8(12)) / ACCEL_SCALE,
                        z: bytesToInt16(data.getUint8(13), data.getUint8(14)) / ACCEL_SCALE
                    }
                };
            }

            function downloadData() {
                if (recordedData.length === 0) {
                    alert('No data to download');
                    return;
                }

                const timestamp = new Date().toISOString()
                    .replace(/[:.]/g, '')
                    .slice(0, -4);

                const rows = ["timestamp,x-axis (g),y-axis (g),z-axis (g),x-axis (deg/s),y-axis (deg/s),z-axis (deg/s)"];
                recordedData.forEach(data => {
                    rows.push(`${data.timestamp},${data.accel.x.toFixed(6)},${data.accel.y.toFixed(6)},${data.accel.z.toFixed(6)},${data.gyro.x.toFixed(6)},${data.gyro.y.toFixed(6)},${data.gyro.z.toFixed(6)}`);
                });

                const blob = new Blob([rows.join('\\n')], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.setAttribute('href', url);
                a.setAttribute('download', `esense_imu_data_${timestamp}.csv`);
                a.click();
                window.URL.revokeObjectURL(url);
            }

            async function connectToESense() {
                try {
                    device = await navigator.bluetooth.requestDevice({
                        filters: [{ namePrefix: 'eSense' }],
                        optionalServices: ['0000ff06-0000-1000-8000-00805f9b34fb']
                    });

                    const server = await device.gatt.connect();
                    const service = await server.getPrimaryService('0000ff06-0000-1000-8000-00805f9b34fb');
                    
                    configCharacteristic = await service.getCharacteristic('0000ff07-0000-1000-8000-00805f9b34fb');
                    imuDataCharacteristic = await service.getCharacteristic('0000ff08-0000-1000-8000-00805f9b34fb');

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
                    recordedData = [];
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
                    recordedData.push(data);
                    document.getElementById('sampleCount').textContent = 
                        `Samples collected: ${recordedData.length}`;
                }

                document.getElementById('dataDisplay').textContent = 
                    `Latest IMU Data:\\n` +
                    `Time: ${data.timestamp}\\n` +
                    `Accelerometer (g): x=${data.accel.x.toFixed(3)}, y=${data.accel.y.toFixed(3)}, z=${data.accel.z.toFixed(3)}\\n` +
                    `Gyroscope (deg/s): x=${data.gyro.x.toFixed(3)}, y=${data.gyro.y.toFixed(3)}, z=${data.gyro.z.toFixed(3)}`;
            }

            // Add button event listeners
            document.getElementById('connectButton').addEventListener('click', connectToESense);
            document.getElementById('startButton').addEventListener('click', startSampling);
            document.getElementById('stopButton').addEventListener('click', stopSampling);
            document.getElementById('downloadButton').addEventListener('click', downloadData);
        </script>
    </div>
    """

# This is for when the file is run directly
if __name__ == "__main__":
    esens_collection_page()