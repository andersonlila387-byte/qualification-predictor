<?php
/**
 * PHP Proxy for Python FastAPI ML Model
 * Transfers multi-part data to the ML engine
 */

header('Content-Type: application/json');

// Disable output buffering to allow streaming
@ini_set('zlib.output_compression', 0);
@ini_set('implicit_flush', 1);
while (ob_get_level()) ob_end_flush();

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['status' => 'error', 'message' => 'Method not allowed']);
    exit;
}

// Use REQUEST to catch action from GET (URL) or POST (Form body)
$action = $_REQUEST['action'] ?? 'predict';

$base_url = 'http://localhost:8000';
$headers = [];
$post_data = [];

$curl = curl_init();

// Route configuration
if ($action === 'evaluate') {
    // Adaptability evaluation requires JSON
    $python_api_url = $base_url . '/adaptability/evaluate';
    $headers[] = 'Content-Type: application/json';
    
    // Ensure answers is an array
    $answers = $_POST['answers'] ?? [];
    if (!is_array($answers)) {
        $answers = explode(',', $answers); // Fallback if sent as string
    }
    $post_data = json_encode(['responses' => $answers]);
    
} else {
    // Predict and Generate Questions use Multipart/Form-Data
    $python_api_url = $base_url . '/' . ($action === 'generate_questions' ? 'generate_questions' : 'predict');

// Prepare form fields
$post_fields = [
    'position' => $_POST['position'] ?? '',
    'company' => $_POST['company'] ?? '',
    'experience' => $_POST['experience'] ?? '',
    'skills' => $_POST['skills'] ?? '',
    // 'answers' is not used by predict/generate endpoints directly
    'adaptability_score' => $_POST['adaptability_score'] ?? '0.5'
];

// Handle File upload
if (isset($_FILES['resume']) && $_FILES['resume']['error'] === UPLOAD_ERR_OK) {
    $post_fields['resume'] = new CURLFile(
        $_FILES['resume']['tmp_name'], 
        $_FILES['resume']['type'], 
        $_FILES['resume']['name']
    );
}
    $post_data = $post_fields;
}

curl_setopt_array($curl, [
    CURLOPT_URL => $python_api_url,
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => $post_data,
    CURLOPT_HTTPHEADER => $headers,
    CURLOPT_RETURNTRANSFER => true, // Enable return transfer
    CURLOPT_TIMEOUT => 300, // Increased timeout for AI generation
]);

$response = curl_exec($curl);
$error = curl_error($curl);

if ($response === false) {
    http_response_code(503);
    echo json_encode(['error' => 'ML Server connection failed', 'details' => $error]);
} else {
    echo $response;
}

curl_close($curl);