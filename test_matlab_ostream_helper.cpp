#include <iostream>
#include <string>

#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"

#include "utils/matlab_ostream_helper.h"

int main() {
  // start MATLAB engine synchronously
  std::unique_ptr<matlab::engine::MATLABEngine>
    matlab_ptr = matlab::engine::startMATLAB();
  std::cout << "MATLAB Engine started." << std::endl;
  matlab::data::ArrayFactory factory;

  auto sqrt = matlab_ptr->feval(u"sqrt",
    factory.createArray({ 1,4 }, { -2.0, 2.0, 6.0, 8.0 }));
  std::cout << sqrt << std::endl;

  const auto s = factory.createScalar(u"dummy_string");
  std::cout << s << std::endl;
  matlab::data::MATLABString str = s[0];
  std::cout << cm::utf16_to_utf8(*str) << std::endl;

  const matlab::data::Array ca = factory.createCellArray({ 1, 2 },
    factory.createCharArray("MATLAB Cell Array"),
    factory.createArray<double>({ 2, 2 }, { 1.2, 2.2, 3.2, 4.2 }));
  std::cout << ca << std::endl;
  std::cout << matlab::data::CellArray(ca) << std::endl;

  auto func = matlab_ptr->feval(u"str2func",
    factory.createCharArray("@(x)3*x(1)^2 + 2*x(1)*x(2) + x(2)^2 - 4*x(1) + 5*x(2)"));
  auto solver = matlab_ptr->feval(u"str2func", factory.createCharArray("@fminunc"));
  auto options = matlab_ptr->feval(u"optimoptions", std::vector<matlab::data::Array>({
    solver, factory.createCharArray("Display"), factory.createCharArray("iter-detailed")}));
  std::vector<matlab::data::Array> args = {
    func,
    factory.createArray<double>({1, 2}, {1.0, 1.0}),
    options };
  auto result = matlab_ptr->feval(u"fminunc", 5, args);
  std::cout << "#func:\n" << func << std::endl;
  std::cout << "#solver:\n" << solver << std::endl;
  std::cout << "#options:\n" << options << std::endl;
  std::cout << "#args:" << std::endl;
  for (const auto &a : args)
    std::cout << a << std::endl;
  std::cout << "#result:" << std::endl;
  for (const auto &a : result)
    std::cout << a << std::endl;

  return 0;
}
