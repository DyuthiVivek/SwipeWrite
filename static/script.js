var socket = io.connect('http://127.0.0.1:5000');


        // Handle incoming video frames
        socket.on('video_frame', function(data) {
            var img = document.getElementById('video_feed');
            img.src = 'data:image/jpeg;base64,' + data.image;
            console.log(data.coordinates, data.draw);
            
            // Start drawing based on server-sent data
            if (data.draw) {
            // Enable drawing based on the condition received from the server
                isDrawing = true;
            } else {
            // Disable drawing
                isDrawing = false;
            }

            // Update drawing coordinates if provided by the server
            if (data.coordinates) {
                [lastX, lastY] = data.coordinates;
            }
        });

        const canvas = document.createElement("canvas");
        document.body.appendChild(canvas);
        const ctx = canvas.getContext("2d");
        const colorPicker = document.getElementById("colorPicker");
        const penButton = document.getElementById("penButton");
        const eraserButton = document.getElementById("eraserButton");
        const rectangleButton = document.getElementById("rectangleButton");
        const circleButton = document.getElementById("circleButton");

        let isDrawing = false;
        let isErasing = false;
        let isDrawingRectangle = false;
        let isDrawingCircle = false;
        let lastX = 0;
        let lastY = 0;
        let shapeStartX = 0;
        let shapeStartY = 0;

        canvas.width = window.innerWidth - 20;
        canvas.height = window.innerHeight - 150;

        canvas.addEventListener("mousedown", (e) => {
            isDrawing = true;
            [lastX, lastY] = [e.offsetX, e.offsetY];
            if (isErasing) ctx.clearRect(lastX - 10, lastY - 10, 20, 20);
        });

        canvas.addEventListener("mousemove", draw);
        canvas.addEventListener("mouseup", () => isDrawing = false);
        canvas.addEventListener("mouseout", () => isDrawing = false);

        penButton.addEventListener("click", () => {
            isErasing = false;
            isDrawingRectangle = false;
            isDrawingCircle = false;
            canvas.style.cursor = "crosshair";
        });

        eraserButton.addEventListener("click", () => {
            isErasing = true;
            isDrawingRectangle = false;
            isDrawingCircle = false;
            canvas.style.cursor = "url('eraser.png'), auto";
        });

        rectangleButton.addEventListener("click", () => {
            isErasing = false;
            isDrawingRectangle = true;
            isDrawingCircle = false;
            canvas.style.cursor = "crosshair";
        });

        circleButton.addEventListener("click", () => {
            isErasing = false;
            isDrawingRectangle = false;
            isDrawingCircle = true;
            canvas.style.cursor = "crosshair";
        });

        colorPicker.addEventListener("input", () => {
            ctx.strokeStyle = colorPicker.value;
            ctx.fillStyle = colorPicker.value;
        });

        function draw(e) {
            if (!isDrawing) return;
        
            ctx.lineWidth = 2;
            ctx.lineCap = "round";
        
            if (isErasing) {
                ctx.clearRect(e.offsetX - 10, e.offsetY - 10, 20, 20);
            } else if (isDrawingRectangle) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.strokeRect(shapeStartX, shapeStartY, e.offsetX - shapeStartX, e.offsetY - shapeStartY);
            } else if (isDrawingCircle) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                const radius = Math.sqrt(Math.pow(e.offsetX - shapeStartX, 2) + Math.pow(e.offsetY - shapeStartY, 2));
                ctx.beginPath();
                ctx.arc(shapeStartX, shapeStartY, radius, 0, 2 * Math.PI);
                ctx.stroke();
            } else {
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(e.offsetX, e.offsetY);
                ctx.stroke();
                // Update lastX and lastY based on the current drawing position
                lastX = e.offsetX;
                lastY = e.offsetY;
            }
        }
