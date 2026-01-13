#ifndef CM_UTF16_TO_UTF8
#define CM_UTF16_TO_UTF8

#include <codecvt>
#include <locale>
#include <string>

namespace cm {

/*!
 * \brief Utility function to convert a std::u16string to a std::string.
 * \param utf16_string input std::u16string
 * \return a std::string
 * \sa https://stackoverflow.com/a/35103224/5487342
 * \sa https://stackoverflow.com/q/41107667/5487342
 * \sa https://stackoverflow.com/q/7232710/5487342
 */

#if _MSC_VER >= 1900

inline std::string utf16_to_utf8(const std::u16string &utf16_string) {
  static std::wstring_convert<std::codecvt_utf8_utf16<int16_t>, int16_t> convert;
  auto p = reinterpret_cast<const int16_t *>(utf16_string.data());
  return convert.to_bytes(p, p + utf16_string.size());
}

#else

inline std::string utf16_to_utf8(const std::u16string &utf16_string) {
  static std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> convert;
  return convert.to_bytes(utf16_string);
}

#endif

} // namespace cm

#endif // CM_UTF16_TO_UTF8
