import express from "express";
import cookieParser from "cookie-parser";
import dotenv from "dotenv";
import bodyParser from "body-parser";
import path from 'path';
import { fileURLToPath } from 'url';
import cors from 'cors';
import connectDB from './config/db.js';

import authRoutes from './routes/authRoutes.js';
import imageRoutes from './routes/imageRoutes.js';



const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


dotenv.config();

// Connect to MongoDB
connectDB()
    .then(() => console.log('MongoDB connected successfully'))
    .catch(err => {
        console.error('MongoDB connection error:', err);
        process.exit(1);
    });

const app = express();
const PORT = process.env.PORT || 4000;

// Middleware setup
app.use(express.json());
app.use(cookieParser());
app.use(cors());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, '../frontend')));

// Routes

app.use('/auth', authRoutes);
app.use("/api", imageRoutes);

// Serve frontend
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ 
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? err.message : undefined
    });
});



// // API URLs configuration (remove duplicate)
// const API_URLS = {
//     sdxl: "https://empty-stingray-42.loca.lt/generate",
//     pixart: "https://myflaskdemo.loca.lt/generate"
// };

// const MODEL_INFO = {
//     sdxl: {
//         name: "SDXL 1.0",
//         description: "Stable Diffusion XL - Higher quality, detailed images"
//     },
//     pixart: {
//         name: "PixArt-α",
//         description: "Fast generation, artistic style"
//     }
// };


// // Image generation endpoint (Flask API returns JSON with { url })
// app.post("/api/generate-image", async (req, res) => {
//     try {
//         const { prompt, model = "sdxl" } = req.body;
//         console.log("Received prompt:", prompt);
//         console.log("Selected model:", model);

//         if (!prompt) {
//             return res.status(400).json({ error: "Prompt is required" });
//         }

//         const apiUrl = API_URLS[model];
//         if (!apiUrl) {
//             return res.status(400).json({ error: `Invalid model: ${model}` });
//         }

//         console.log(`Calling ${model.toUpperCase()} API:`, apiUrl);

//         // Call the AI API (Flask) which returns JSON { url: ... }
//         const response = await fetch(apiUrl, {
//             method: "POST",
//             headers: {
//                 "Content-Type": "application/json",
//                 "Accept": "application/json",
//                 "ngrok-skip-browser-warning": "true"
//             },
//             body: JSON.stringify({ prompt })
//         });

//         if (!response.ok) {
//             const errorText = await response.text();
//             console.error("AI API Error:", errorText);
//             throw new Error(`AI API responded with status ${response.status}`);
//         }

//         const data = await response.json();

//         if (!data.url) {
//             throw new Error("AI API response does not contain url");
//         }

//         // Forward to frontend as imageUrl
//         res.json({
//             imageUrl: data.url,
//             prompt,
//             model
//         });

//     } catch (err) {
//         console.error("Image generation error:", err);
//         res.status(500).json({
//             error: err.message,
//             details: process.env.NODE_ENV === "development" ? err.stack : undefined
//         });
//     }
// });




// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});