/**
 * Smart PPE Compliance Detection System - Global Client Script
 */

document.addEventListener("DOMContentLoaded", () => {
    console.log("Smart PPE Compliance Detection UI Loaded.");
    
    // Check hardware status on top bar
    checkHardwareHealth();
});

async function checkHardwareHealth() {
    try {
        const res = await fetch("/api/camera/status");
        const data = await res.json();
        const topHw = document.getElementById("topHwStatus");
        if (topHw && data && data.hardware) {
            if (data.hardware.connected) {
                topHw.textContent = `ONLINE (${data.hardware.port})`;
                topHw.style.color = "#2ecc71";
            } else {
                topHw.textContent = `STANDBY (${data.hardware.port})`;
                topHw.style.color = "#f39c12";
            }
        }
    } catch (e) {
        // Quiet fail
    }
}
