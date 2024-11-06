import torch
import time
from homework.models import Detector, Classifier

def debug_inference_time():
    print("\n=== Testing Inference Time ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Detector().to(device)
    model.load_state_dict(torch.load('homework/detector.th', map_location=device))
    model.eval()
    
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 96, 128).to(device)
    
    warmup_rounds = 10
    test_rounds = 100
    
    print("Warming up...")
    with torch.inference_mode():
        for _ in range(warmup_rounds):
            _ = model.predict(dummy_input)
    
    print("Testing inference time...")
    times = []
    with torch.inference_mode():
        for _ in range(test_rounds):
            start = time.time()
            _ = model.predict(dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Max inference time: {max_time*1000:.2f}ms")
    print(f"Time for 100 samples: {avg_time*100*1000:.2f}ms")

def debug_memory():
    print("\n=== Testing Memory Usage ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    
    model = Detector().to(device)
    model.load_state_dict(torch.load('homework/detector.th', map_location=device))
    
    if torch.cuda.is_available():
        print(f"After model load: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 96, 128).to(device)
    
    with torch.inference_mode():
        outputs = model.predict(dummy_input)
        if torch.cuda.is_available():
            print(f"After inference: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
        del outputs
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"After cleanup: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    
    print("\nModel parameter count:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

def debug_accuracy_test():
    print("\n=== Testing Accuracy Computation Time ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nTesting Detector:")
    detector = Detector().to(device)
    detector.load_state_dict(torch.load('homework/detector.th', map_location=device, weights_only=True))
    detector.eval()
    
    print("\nTesting Classifier:")
    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load('homework/classifier.th', map_location=device, weights_only=True))
    classifier.eval()
    
    # Test different batch sizes
    batch_sizes = [1, 8, 16, 32, 64]
    input_shape_detector = (96, 128)
    input_shape_classifier = (64, 64)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Test Detector
        test_input = torch.randn(batch_size, 3, *input_shape_detector).to(device)
        with torch.inference_mode():
            start = time.time()
            for _ in range(100):  # Test 100 batches
                pred, _ = detector.predict(test_input)
            torch.cuda.synchronize()
            end = time.time()
        print(f"Detector time for 100 batches: {(end-start)*1000:.2f}ms")
        
        # Test Classifier
        test_input = torch.randn(batch_size, 3, *input_shape_classifier).to(device)
        with torch.inference_mode():
            start = time.time()
            for _ in range(100):  # Test 100 batches
                pred = classifier.predict(test_input)
            torch.cuda.synchronize()
            end = time.time()
        print(f"Classifier time for 100 batches: {(end-start)*1000:.2f}ms")
        
    # Test memory clearing
    print("\nTesting memory management:")
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"Memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    

if __name__ == '__main__':
    debug_inference_time()
    debug_memory()
    debug_accuracy_test()