class Camera {
    constructor(videoElement) {
        this.videoElement = videoElement;
        this.stream = null;
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            
            this.videoElement.srcObject = this.stream;
            return new Promise((resolve) => {
                this.videoElement.onloadedmetadata = () => {
                    this.videoElement.play();
                    resolve();
                };
            });
        } catch (error) {
            console.error('Error starting camera:', error);
            throw error;
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }

    capture(canvasElement) {
        const context = canvasElement.getContext('2d');
        
        // Set canvas dimensions to video dimensions
        canvasElement.width = this.videoElement.videoWidth;
        canvasElement.height = this.videoElement.videoHeight;
        
        // Draw current video frame to canvas
        context.drawImage(this.videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        // Return data URL of the captured image
        return canvasElement.toDataURL('image/jpeg');
    }

    captureAsBlob(canvasElement) {
        return new Promise((resolve) => {
            const context = canvasElement.getContext('2d');
            
            // Set canvas dimensions to video dimensions
            canvasElement.width = this.videoElement.videoWidth;
            canvasElement.height = this.videoElement.videoHeight;
            
            // Draw current video frame to canvas
            context.drawImage(this.videoElement, 0, 0, canvasElement.width, canvasElement.height);
            
            // Convert canvas to blob
            canvasElement.toBlob((blob) => {
                resolve(blob);
            }, 'image/jpeg', 0.95);
        });
    }

    async captureAndSubmit(canvasElement, fileInput, handleFilesCallback) {
        const blob = await this.captureAsBlob(canvasElement);
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        
        // Create a FileList-like object
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        
        // Call the file handling function
        handleFilesCallback(fileInput.files);
    }
}

// Example usage:
// const camera = new Camera(document.getElementById('camera-preview'));
// camera.start().then(() => console.log('Camera started'));
// document.getElementById('capture-button').addEventListener('click', () => {
//     const canvas = document.createElement('canvas');
//     camera.captureAndSubmit(canvas, document.getElementById('file-input'), handleFiles);
// });