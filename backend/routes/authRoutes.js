import express from 'express';
import { register, login, logout } from '../controllers/authController.js';
import { requireAuth } from '../middleware/authMiddleware.js';
import User from '../models/User.js';

const router = express.Router();

// Auth routes
router.post('/register', register);
router.post('/login', login);
router.post('/logout', requireAuth, logout);

// Get current user session
router.get('/session', requireAuth, async (req, res) => {
    try {
        const user = await User.findById(req.user._id).select('-password');
        res.json({ user });
    } catch (error) {
        res.status(500).json({ message: 'Error fetching session' });
    }
});

// Get user by ID with error logging
router.get('/user/:id', requireAuth, async (req, res) => {
    try {
        console.log('Fetching user with ID:', req.params.id);
        
        const user = await User.findById(req.params.id).select('-password');
        console.log('User found:', user);
        
        if (!user) {
            console.log('No user found with ID:', req.params.id);
            return res.status(404).json({ message: 'User not found' });
        }
        
        res.json(user);
    } catch (error) {
        console.error('Error fetching user:', error);
        res.status(500).json({ 
            message: 'Error fetching user',
            error: error.message 
        });
    }
});

export default router;
