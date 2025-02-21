import os , sys


class CustomException(Exception):
    def __init__(self,error_message:Exception,error_details:sys):
        self.error_message = CustomException.get_detailed_error_message(error_message=error_message,error_details=error_details)

    @staticmethod
    def get_detailed_error_message(error_message:Exception,error_details:sys)->str:
        _,_,exec_tb = error_details.exc_info()
        exception_block_line_number = exec_tb.tb_frame.f_lineno #exec_tb go in to the line number where the error occurred
        try_block_line_number = exec_tb.tb_lineno #line by line execution 
        file_name = exec_tb.tb_frame.f_code.co_filename #file name where the error occurred
        error_message = f"Error occurred in script: [{file_name}] at line number: [{try_block_line_number}] and exception block line number: [{exception_block_line_number}] error message: [{error_message}]"
        return error_message
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self)->str:
        return CustomException.__name__.str()