// Test the new Hugging Face Inference Providers API
// Run with: HUGGINGFACE_API_TOKEN="your_token" node test_new_hf_api.js

async function testNewHuggingFaceAPI() {
  const token = process.env.HUGGINGFACE_API_TOKEN;
  if (!token) {
    console.error('❌ HUGGINGFACE_API_TOKEN environment variable is required');
    return;
  }
  
  const API_URL = "https://router.huggingface.co/hf-inference/v3/openai/chat/completions";
  
  try {
    console.log('🧪 Testing NEW Hugging Face Inference Providers API...');
    console.log('🔑 Token:', token.substring(0, 10) + '...');
    console.log('🌐 URL:', API_URL);
    
    const response = await fetch(API_URL, {
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      method: "POST",
      body: JSON.stringify({
        model: "microsoft/Phi-3.5-mini-instruct",
        messages: [
          {
            role: "system",
            content: "You are a helpful assistant."
          },
          {
            role: "user",
            content: "Hello! Can you analyze this test message?"
          }
        ],
        max_tokens: 50,
        temperature: 0.3,
        stream: false
      }),
    });
    
    console.log('📊 Response Status:', response.status);
    console.log('📋 Response Headers:', Object.fromEntries(response.headers.entries()));
    
    const text = await response.text();
    console.log('📄 Response Body:', text);
    
    if (response.ok) {
      try {
        const json = JSON.parse(text);
        console.log('✅ Success! AI Response:', json.choices?.[0]?.message?.content);
      } catch (e) {
        console.log('⚠️ Response was OK but not valid JSON');
      }
    } else {
      console.log('❌ Failed!');
    }
    
  } catch (error) {
    console.error('💥 Error:', error);
  }
}

// Check token permissions
async function checkTokenPermissions() {
  const token = process.env.HUGGINGFACE_API_TOKEN;
  if (!token) {
    console.error('❌ HUGGINGFACE_API_TOKEN environment variable is required');
    return;
  }
  
  try {
    console.log('\n🔍 Checking token permissions...');
    const response = await fetch('https://huggingface.co/api/whoami', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    
    const data = await response.text();
    console.log('Token info:', data);
    
  } catch (error) {
    console.error('Token check error:', error);
  }
}

async function main() {
  await checkTokenPermissions();
  await testNewHuggingFaceAPI();
}

main(); 