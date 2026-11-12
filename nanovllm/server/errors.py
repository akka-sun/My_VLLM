class APIError(Exception):
    def __init__(self, message, status_code=400, error_type='invalid_request_error', param=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = {'error': {'message': message, 'type': error_type, 'param': param, 'code': None}}
