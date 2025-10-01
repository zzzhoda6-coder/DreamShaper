async function checkAuthStatus() {
    try {
        const response = await fetch('/auth/user');
        const data = await response.json();

        if (data.user) {
            // Update UI for logged-in state
            updateAuthUI(true, data.user);
            return data.user;
        } else {
            updateAuthUI(false);
            return null;
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        updateAuthUI(false);
        return null;
    }
}

function updateAuthUI(isLoggedIn, user = null) {
    const loginButton = document.getElementById('loginButton');
    const userAvatar = document.getElementById('userAvatar');
    const userEmail = document.getElementById('userEmail');

    if (isLoggedIn && user) {
        loginButton.style.display = 'none';
        userAvatar.style.display = 'block';
        userEmail.textContent = user.email;
    } else {
        loginButton.style.display = 'flex';
        userAvatar.style.display = 'none';
    }
}

document.getElementById('signupForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        username: document.getElementById('username').value,
        email: document.getElementById('email').value,
        password: document.getElementById('password').value
    };

    console.log('Sending signup request:', formData);

    try {
        const response = await fetch('/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();
        console.log('Signup response:', data);

        if (!response.ok) {
            throw new Error(data.message || 'Signup failed');
        }

        // Store token
        localStorage.setItem('token', data.token);
        
        // Redirect to main page
        window.location.href = '/';
        
    } catch (error) {
        console.error('Signup error:', error);
        // Show error to user
        document.getElementById('errorMessage').textContent = error.message;
    }
});