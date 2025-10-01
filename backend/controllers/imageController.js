import fetch from "node-fetch"; 
import User from '../models/User.js';

const API_URLS = {
  sdxl: "https://f2e3836091f5.ngrok-free.app/api/generate",
  Flux: "https://5607009ea47c.ngrok-free.app/api/generate",
  pixart: "https://16c04af07c75.ngrok-free.app/api/generate",
  Deliberate:"https://deliberate-v2.loca.lt/api/generate"
};

export const generateImage = async (req, res) => {
  try {
    const { prompt, model = "sdxl" } = req.body;
    const userId = req.user._id;

    if (!prompt) {
      return res.status(400).json({ error: "Prompt is required" });
    }

    const apiUrl = API_URLS[model];
    if (!apiUrl) {
      return res.status(400).json({ error: `Invalid model: ${model}` });
    }

    // Add ngrok-skip-browser-warning header
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "ngrok-skip-browser-warning": "true"
      },
      body: JSON.stringify({ prompt })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`AI API responded with status ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    
    // Create new image object
    const newImage = {
      prompt: prompt,
      url: data.imageUrl, 
      createdAt: new Date()
    };

    // Save to user's images array
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    user.images.push(newImage);
    await user.save();

    // Send back the image data
    res.status(200).json({
      success: true,
      data: newImage
    });

  } catch (err) {
    console.error("Image generation error:", err);
    res.status(500).json({
      error: err.message || 'Image generation failed'
    });
  }
};





// ----------------- Get all images -----------------
export const getUserImages = async (req, res) => {
  try {
    console.log('Getting images for user:', req.user._id); // for Debug log

    const user = await User.findById(req.user._id)
      .select('images') // Explicitly select the images field
      .sort({ 'images.createdAt': -1 }); // Sort by creation date ترتيب

    if (!user) {
      console.log('User not found in database'); // Debug log
      return res.status(404).json({ 
        success: false,
        error: 'User not found' 
      });
    }

    console.log('Found images:', user.images.length); // Debug log

    return res.status(200).json({
      success: true,
      data: user.images
    });

  } catch (error) {
    console.error('Error in getUserImages:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to fetch images',
      details: error.message
    });
  }
};
