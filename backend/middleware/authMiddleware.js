import jwt from 'jsonwebtoken';
import User from '../models/User.js';

export const requireAuth = async (req, res, next) => {
    try {
        const authHeader = req.headers.authorization;
        console.log('Auth header:', authHeader); // Debug log

        if (!authHeader || !authHeader.startsWith("Bearer ")) {
            return res.status(401).json({ error: "No token provided" });
        }

        const token = authHeader.split(" ")[1];
        console.log('Token extracted'); // Debug log

        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        console.log('Token verified:', decoded); // Debug log
        
        
        const user = await User.findById(decoded.id);
        console.log('Looking for user with ID:', decoded.id);
        
        if (!user) {
            console.log('User not found for ID:', decoded.id);
            return res.status(401).json({ error: "User not found" });
        }

        console.log('User found:', user.username);
        req.user = user;
        next();
    } catch (err) {
        console.error('Auth middleware error:', err);
        return res.status(401).json({ error: "Invalid token" });
    }
};
