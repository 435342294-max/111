#include "utils/Logger.h"

int main() {
  cm::initialize_logger(cm::severity_level::info, "./log");
  LOG_TRACE << "This is a trace message!";
  LOG_DEBUG << "This is a debug message!";
  LOG_INFO << "This is an info message!";
  LOG_WARNING << "This is a warning message!";
  LOG_ERROR << "This is an error message!";
  LOG_FATAL << "This is a fatal message!\n";

  cm::set_severity_level(cm::severity_level::trace);
  LOG_TRACE << "This is a trace message!";
  LOG_DEBUG << "This is a debug message!";
  LOG_INFO << "This is an info message!";
  LOG_WARNING << "This is a warning message!";
  LOG_ERROR << "This is an error message!";
  LOG_FATAL << "This is a fatal message!\n";

  cm::set_severity_level(cm::severity_level::warning);
  LOG_TRACE << "This is a trace message!";
  LOG_DEBUG << "This is a debug message!";
  LOG_INFO << "This is an info message!";
  LOG_WARNING << "This is a warning message!";
  LOG_ERROR << "This is an error message!";
  LOG_FATAL << "This is a fatal message!\n";

  return 0;
}
