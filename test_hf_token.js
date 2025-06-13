// Test script to validate Hugging Face API token
// Run with: node test_hf_token.js

const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

async function testHuggingFaceToken(token) {
  try {
    console.log('Testing Hugging Face API token...');
    
    const response = await fetch('https://api-inference.huggingface.co/models/jackal79/tle-orbit-explainer', {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      method: 'POST',
      body: JSON.stringify({
        inputs: "Test prompt for TLE analysis",
        parameters: {
          max_new_tokens: 50,
          temperature: 0.3
        }
      }),
    });

    console.log('Response status:', response.status);
    
    if (response.ok) {
      console.log('✅ Token is valid! You can use this token.');
      const result = await response.json();
      console.log('Sample response:', result);
      
      console.log('\n🎯 Next steps:');
      console.log('1. Set your environment variable:');
      console.log(`   export HUGGINGFACE_API_TOKEN="${token}"`);
      console.log('2. Or add to your shell profile:');
      console.log(`   echo 'export HUGGINGFACE_API_TOKEN="${token}"' >> ~/.zshrc`);
      console.log('3. Restart your frontend server');
      
    } else {
      const errorText = await response.text();
      console.log('❌ Token validation failed');
      console.log('Error:', errorText);
      
      if (response.status === 401) {
        console.log('\n💡 This means the token is invalid or expired.');
        console.log('Please get a new token from: https://huggingface.co/settings/tokens');
      } else if (response.status === 403) {
        console.log('\n💡 This means you may need access to the model.');
        console.log('Make sure you have access to jackal79/tle-orbit-explainer');
      }
    }
  } catch (error) {
    console.error('❌ Error testing token:', error.message);
  }
}

rl.question('Enter your Hugging Face API token: ', (token) => {
  if (!token || token.trim().length === 0) {
    console.log('❌ No token provided. Get one from: https://huggingface.co/settings/tokens');
    rl.close();
    return;
  }
  
  testHuggingFaceToken(token.trim())
    .finally(() => rl.close());
});

console.log('🔑 Hugging Face Token Validator');
console.log('--------------------------------');
console.log('This will test your API token with the jackal79/tle-orbit-explainer model');
console.log('Get a token from: https://huggingface.co/settings/tokens\n'); 