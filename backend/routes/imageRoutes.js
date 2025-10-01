
import express from "express";
import { generateImage, getUserImages } from "../controllers/imageController.js";
import { requireAuth } from '../middleware/authMiddleware.js';

const router = express.Router();

router.use((err, req, res, next) => {
  console.error('Route error:', err);
  res.status(500).json({ 
    error: err.message || "Internal server error" 
  });
});


router.post("/generate", requireAuth, generateImage);
router.get("/images", requireAuth, getUserImages);

export default router; 


