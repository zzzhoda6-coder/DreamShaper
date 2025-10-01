document.addEventListener('DOMContentLoaded', () => {
    const signupForm = document.getElementById('signupForm');
    const errorMessage = document.getElementById('errorMessage');

    // Clear any existing auth data
    localStorage.removeItem('token');
    localStorage.removeItem('user');

    // Handle form submission
    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        errorMessage.style.display = 'none';

        const formData = {
            username: document.getElementById('username').value.trim(),
            email: document.getElementById('email').value.trim(),
            password: document.getElementById('password').value
        };

        try {
            const response = await fetch('/auth/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            // Check for error responses
            if (!response.ok) {
                throw new Error(data.message || 'Registration failed');
            }

            // Check for success response structure
            if (data.success && data.token && data.user) {
                localStorage.setItem('token', data.token);
                localStorage.setItem('user', JSON.stringify(data.user));
                window.location.href = '/email-confirmed.html';
            } else {
                throw new Error('Invalid response from server');
            }

        } catch (error) {
            errorMessage.textContent = error.message;
            errorMessage.style.display = 'block';
            console.error('Signup error:', error);
        }
    });

    // Password visibility toggle
    document.querySelector('.toggle-password').addEventListener('click', function() {
        const passwordInput = document.getElementById('password');
        const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
        passwordInput.setAttribute('type', type);
        this.classList.toggle('fa-eye');
        this.classList.toggle('fa-eye-slash');
    });
});