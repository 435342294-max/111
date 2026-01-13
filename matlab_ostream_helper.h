#ifndef MATLAB_OSTREAM_HELPER_H
#define MATLAB_OSTREAM_HELPER_H

#include <ostream>
#include <iterator>
#include <algorithm>
#include <complex>

#include "MatlabDataArray.hpp"

#include "utf16_to_utf8.h"

namespace std {

/*!
 * \brief Stream matlab::data::ArrayType object to std::ostream.
 * \param os the output std::ostream
 * \param at the matlab::data::ArrayType object
 * \return the std::ostream
 * \sa https://stackoverflow.com/a/48478218/5487342
 */
inline std::ostream &operator<<(std::ostream &os, const matlab::data::ArrayType &at) {
  using namespace matlab::data;
  switch (at) {
    case ArrayType::LOGICAL:               return os << "LOGICAL";
    case ArrayType::CHAR:                  return os << "CHAR";
    case ArrayType::MATLAB_STRING:         return os << "MATLAB_STRING";
    case ArrayType::DOUBLE:                return os << "DOUBLE";
    case ArrayType::SINGLE:                return os << "SINGLE";
    case ArrayType::INT8:                  return os << "INT8";
    case ArrayType::UINT8:                 return os << "UINT8";
    case ArrayType::INT16:                 return os << "INT16";
    case ArrayType::UINT16:                return os << "UINT16";
    case ArrayType::INT32:                 return os << "INT32";
    case ArrayType::UINT32:                return os << "UINT32";
    case ArrayType::INT64:                 return os << "INT64";
    case ArrayType::UINT64:                return os << "UINT64";
    case ArrayType::COMPLEX_DOUBLE:        return os << "COMPLEX_DOUBLE";
    case ArrayType::COMPLEX_SINGLE:        return os << "COMPLEX_SINGLE";
    case ArrayType::COMPLEX_INT8:          return os << "COMPLEX_INT8";
    case ArrayType::COMPLEX_UINT8:         return os << "COMPLEX_UINT8";
    case ArrayType::COMPLEX_INT16:         return os << "COMPLEX_INT16";
    case ArrayType::COMPLEX_UINT16:        return os << "COMPLEX_UINT16";
    case ArrayType::COMPLEX_INT32:         return os << "COMPLEX_INT32";
    case ArrayType::COMPLEX_UINT32:        return os << "COMPLEX_UINT32";
    case ArrayType::COMPLEX_INT64:         return os << "COMPLEX_INT64";
    case ArrayType::COMPLEX_UINT64:        return os << "COMPLEX_UINT64";
    case ArrayType::CELL:                  return os << "CELL";
    case ArrayType::STRUCT:                return os << "STRUCT";
    case ArrayType::OBJECT:                return os << "OBJECT";
    case ArrayType::VALUE_OBJECT:          return os << "VALUE_OBJECT";
    case ArrayType::HANDLE_OBJECT_REF:     return os << "HANDLE_OBJECT_REF";
    case ArrayType::ENUM:                  return os << "ENUM";
    case ArrayType::SPARSE_LOGICAL:        return os << "SPARSE_LOGICAL";
    case ArrayType::SPARSE_DOUBLE:         return os << "SPARSE_DOUBLE";
    case ArrayType::SPARSE_COMPLEX_DOUBLE: return os << "SPARSE_COMPLEX_DOUBLE";
    default:                               return os << "UNKNOWN";
  }
}

/*!
 * \brief Stream matlab::data::ArrayDimensions object to std::ostream.
 * \param os the output std::ostream
 * \param d the matlab::data::ArrayDimensions object
 * \return the std::ostream
 */
inline std::ostream &operator<<(std::ostream &os, const matlab::data::ArrayDimensions &d) {
  if (!d.empty()) {
    std::copy(d.begin(), std::prev(d.end()), std::ostream_iterator<std::size_t>(os, " x "));
    os << d.back();
  }
  return os;
}

/*!
 * \brief Stream matlab::data::Array object to std::ostream.
 * Forward declared for recursive invocation in Struct and Cell.
 * \param os the output std::ostream
 * \param a the matlab::data::Array object
 * \return the std::ostream
 */
inline std::ostream &operator<<(std::ostream &os, const matlab::data::Array &a);

/*!
 * \brief Stream matlab::data::MATLABString object to std::ostream.
 * \param os the output std::ostream
 * \param ms the matlab::data::MATLABString object
 * \return the std::ostream
 */
inline std::ostream &operator<<(std::ostream &os, const matlab::data::MATLABString &ms) {
  if (ms.has_value())
    os << cm::utf16_to_utf8(*ms);
  return os;
}

/*!
 * \brief Stream matlab::data::MATLABString object to std::ostream.
 * \param os the output std::ostream
 * \param ms the matlab::data::MATLABString object
 * \return the std::ostream
 */
inline std::ostream &operator<<(std::ostream &os, const matlab::data::StringArray &sa) {
  os << sa.getDimensions() << " string array";
  for (const auto &s : sa)
    os << '\n' << s;
  return os;
}

/*!
 * \brief Stream matlab::data::CharArray object to std::ostream.
 * \param os the output std::ostream
 * \param ca the matlab::data::CharArray object
 * \return the std::ostream
 */
inline std::ostream &operator<<(std::ostream &os, const matlab::data::CharArray &ca) {
  return os << ca.toAscii();
}

/*!
 * \brief Stream matlab::data::CellArray object to std::ostream.
 * \param os the output std::ostream
 * \param s the matlab::data::CellArray object
 * \return the std::ostream
 */
inline std::ostream &operator<<(std::ostream &os, const matlab::data::CellArray &ca) {
  os << ca.getDimensions() << " cell array";
  for (const auto &a : ca)
    os << '\n' << a;
  return os;
}

/*!
 * \brief Stream matlab::data::StructArray object to std::ostream in the MATLAB style.
 * \param os the output std::ostream
 * \param sa the matlab::data::StructArray object
 * \return the std::ostream
 */
inline std::ostream &operator<<(std::ostream &os, const matlab::data::StructArray &sa) {
  std::vector<std::string> fields;
  for (const auto &f : sa.getFieldNames())
    fields.push_back(f);
  if (fields.empty())
    return os << "struct array with no fields";
  os << sa.getDimensions() << " struct array with fields: ";
  for (const auto &f : fields)
    os << '\n' << f;
  for (const auto &s : sa) {
    for (const auto &fn : fields)
      os << '\n' << fn << ":\n" << s[fn];
  }
  return os;
}

/*!
 * \brief Stream a typed matlab::data::Array object to std::ostream.
 * \tparam T array type
 * \param os the output std::ostream
 * \param a the matlab::data::Array object
 * \return the std::ostream
 */
template <typename T>
std::ostream &typed_array_to_ostream(std::ostream &os, const matlab::data::Array &a) {
  os << a.getDimensions() << ' ' << a.getType() << " array";
  matlab::data::TypedArray<T> ta = a;
  if (!ta.isEmpty()) {
    os << '\n';
    std::copy(ta.begin(), ta.end(), std::ostream_iterator<T>(os, " "));
  }
  return os;
}

/*!
 * \brief Stream matlab::data::Array object to std::ostream.
 * \param os the output std::ostream
 * \param a the matlab::data::Array object
 * \return the std::ostream
 */
inline std::ostream &operator<<(std::ostream &os, const matlab::data::Array &a) {
  using namespace matlab::data;
  switch (a.getType()) {
    case ArrayType::LOGICAL:       return typed_array_to_ostream<bool>(os, a);
    case ArrayType::CHAR:          return os << CharArray(a);
    case ArrayType::MATLAB_STRING: return os << StringArray(a);
    case ArrayType::DOUBLE:        return typed_array_to_ostream<double>(os, a);
    case ArrayType::SINGLE:        return typed_array_to_ostream<float>(os, a);
    case ArrayType::INT8:          return typed_array_to_ostream<int8_t>(os, a);
    case ArrayType::UINT8:         return typed_array_to_ostream<uint8_t>(os, a);
    case ArrayType::INT16:         return typed_array_to_ostream<int16_t>(os, a);
    case ArrayType::UINT16:        return typed_array_to_ostream<uint16_t>(os, a);
    case ArrayType::INT32:         return typed_array_to_ostream<int32_t>(os, a);
    case ArrayType::UINT32:        return typed_array_to_ostream<uint32_t>(os, a);
    case ArrayType::INT64:         return typed_array_to_ostream<int64_t>(os, a);
    case ArrayType::UINT64:        return typed_array_to_ostream<uint64_t>(os, a);
    case ArrayType::COMPLEX_DOUBLE:
      return typed_array_to_ostream<std::complex<double>>(os, a);
    case ArrayType::COMPLEX_SINGLE:
      return typed_array_to_ostream<std::complex<float>>(os, a);
    case ArrayType::COMPLEX_INT8:
      return typed_array_to_ostream<std::complex<int8_t>>(os, a);
    case ArrayType::COMPLEX_UINT8:
      return typed_array_to_ostream<std::complex<uint8_t>>(os, a);
    case ArrayType::COMPLEX_INT16:
      return typed_array_to_ostream<std::complex<int16_t>>(os, a);
    case ArrayType::COMPLEX_UINT16:
      return typed_array_to_ostream<std::complex<uint16_t>>(os, a);
    case ArrayType::COMPLEX_INT32:
      return typed_array_to_ostream<std::complex<int32_t>>(os, a);
    case ArrayType::COMPLEX_UINT32:
      return typed_array_to_ostream<std::complex<uint32_t>>(os, a);
    case ArrayType::COMPLEX_INT64:
      return typed_array_to_ostream<std::complex<int64_t>>(os, a);
    case ArrayType::COMPLEX_UINT64:
      return typed_array_to_ostream<std::complex<uint64_t>>(os, a);
    case ArrayType::CELL:          return os << CellArray(a);
    case ArrayType::STRUCT:        return os << StructArray(a);
    default:
      return os << a.getDimensions() << ' ' << a.getType() << " array";
  }
}

} // namespace std

#endif // MATLAB_OSTREAM_HELPER_H
