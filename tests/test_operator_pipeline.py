"""
Test script for the complete Operator Layer pipeline.

Tests end-to-end processing, error scenarios, and recovery mechanisms.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from foodvision_ai.operator.pipeline_integration import (
    IntegratedPipeline, 
    test_pipeline_with_sample_images,
    test_error_scenarios,
    run_complete_pipeline_test
)
from loguru import logger


async def test_basic_pipeline_functionality():
    """Test basic pipeline functionality with a simple image."""
    logger.info("Testing basic pipeline functionality")
    
    pipeline = IntegratedPipeline()
    
    # Test with a simple food image URL
    test_image_url = "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=500"
    
    try:
        result = await pipeline.process_image_complete(
            image_id="basic_test",
            image_url=test_image_url,
            metadata={"test_type": "basic_functionality"}
        )
        
        logger.info(f"Basic test result: {result['success']}")
        
        # Check that we got some results
        assert "processing_result" in result
        assert "status" in result
        assert "session_summary" in result
        
        if result["success"]:
            logger.success("Basic pipeline functionality test PASSED")
        else:
            logger.warning("Basic pipeline functionality test completed with errors")
        
        return result
        
    except Exception as e:
        logger.error(f"Basic pipeline functionality test FAILED: {e}")
        return {"success": False, "error": str(e)}


async def test_status_tracking():
    """Test status tracking throughout the pipeline."""
    logger.info("Testing status tracking")
    
    pipeline = IntegratedPipeline()
    
    # Test with a simple image
    test_image_url = "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=500"
    
    try:
        # Start processing
        processing_task = asyncio.create_task(
            pipeline.process_image_complete(
                image_id="status_test",
                image_url=test_image_url,
                metadata={"test_type": "status_tracking"}
            )
        )
        
        # Check status during processing
        await asyncio.sleep(0.1)  # Give it a moment to start
        
        status = pipeline.get_processing_status("status_test")
        if status:
            logger.info(f"Status during processing: {status['status']['current_phase']}")
        
        # Wait for completion
        result = await processing_task
        
        # Check final status
        final_status = pipeline.get_processing_status("status_test")
        if final_status:
            logger.info(f"Final status: {final_status['status']['current_phase']}")
            logger.info(f"Progress: {final_status['status']['progress_percentage']}%")
        
        logger.success("Status tracking test PASSED")
        return result
        
    except Exception as e:
        logger.error(f"Status tracking test FAILED: {e}")
        return {"success": False, "error": str(e)}


async def test_logging_functionality():
    """Test logging functionality."""
    logger.info("Testing logging functionality")
    
    pipeline = IntegratedPipeline()
    
    # Test with a simple image
    test_image_url = "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=500"
    
    try:
        result = await pipeline.process_image_complete(
            image_id="logging_test",
            image_url=test_image_url,
            metadata={"test_type": "logging_functionality"}
        )
        
        # Check session logs
        logs = pipeline.get_session_logs("logging_test")
        
        assert "session_summary" in logs
        assert "remarks" in logs
        
        logger.info(f"Session generated {len(logs['remarks'])} log remarks")
        logger.info(f"Session summary: {logs['session_summary']['total_remarks']} total remarks")
        
        # Check that we have logs from different components
        components = set(remark["component"] for remark in logs["remarks"])
        logger.info(f"Components that logged: {components}")
        
        logger.success("Logging functionality test PASSED")
        return result
        
    except Exception as e:
        logger.error(f"Logging functionality test FAILED: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Run all operator pipeline tests."""
    logger.info("Starting Operator Layer Pipeline Tests")
    
    test_results = []
    
    # Test 1: Basic functionality
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Basic Pipeline Functionality")
    logger.info("="*50)
    basic_result = await test_basic_pipeline_functionality()
    test_results.append(("Basic Functionality", basic_result["success"]))
    
    # Test 2: Status tracking
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Status Tracking")
    logger.info("="*50)
    status_result = await test_status_tracking()
    test_results.append(("Status Tracking", status_result["success"]))
    
    # Test 3: Logging functionality
    logger.info("\n" + "="*50)
    logger.info("TEST 3: Logging Functionality")
    logger.info("="*50)
    logging_result = await test_logging_functionality()
    test_results.append(("Logging Functionality", logging_result["success"]))
    
    # Test 4: Sample images test
    logger.info("\n" + "="*50)
    logger.info("TEST 4: Sample Images Pipeline Test")
    logger.info("="*50)
    try:
        sample_results = await test_pipeline_with_sample_images()
        sample_success = sample_results["success_rate"] > 0.5  # At least 50% success
        test_results.append(("Sample Images", sample_success))
        logger.info(f"Sample images test: {sample_results['successful_tests']}/{sample_results['total_tests']} passed")
    except Exception as e:
        logger.error(f"Sample images test failed: {e}")
        test_results.append(("Sample Images", False))
    
    # Test 5: Error scenarios
    logger.info("\n" + "="*50)
    logger.info("TEST 5: Error Scenarios")
    logger.info("="*50)
    try:
        error_results = await test_error_scenarios()
        error_success = error_results["gracefully_handled"] == error_results["total_error_tests"]
        test_results.append(("Error Scenarios", error_success))
        logger.info(f"Error scenarios test: {error_results['gracefully_handled']}/{error_results['total_error_tests']} handled gracefully")
    except Exception as e:
        logger.error(f"Error scenarios test failed: {e}")
        test_results.append(("Error Scenarios", False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed_tests = sum(1 for _, success in test_results if success)
    total_tests = len(test_results)
    
    for test_name, success in test_results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.success("All Operator Layer Pipeline tests PASSED!")
        return True
    else:
        logger.error(f"Some tests failed. {total_tests - passed_tests} tests need attention.")
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)