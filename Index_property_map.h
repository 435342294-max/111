#ifndef CM_INDEX_PROPERTY_MAP_H
#define CM_INDEX_PROPERTY_MAP_H

#include <boost/property_map/property_map.hpp>

namespace cm {

/*!
 * \brief An adaptor to turn a SequenceContainer like
 * std::vector or std::array into an Lvalue Property Map.
 * \tparam SequenceContainer A SequenceContainer is a Container that
 * stores objects of the same type in a linear arrangement.
 * \todo sequence is an array or pointer
 */
template <typename SequenceContainer>
class Index_property_map
  : public boost::put_get_helper<
    typename SequenceContainer::value_type &,
    Index_property_map<SequenceContainer>> {
public:
  // typedefs
  typedef typename SequenceContainer::size_type key_type;
  typedef typename SequenceContainer::value_type value_type;
  typedef value_type &reference;
  typedef typename boost::lvalue_property_map_tag category;

  Index_property_map() : m_psc(nullptr) {}

  Index_property_map(SequenceContainer &sc) : m_psc(&sc) {}

  reference operator[](const key_type &k) const {
    return (*m_psc)[k];
  }

private:
  SequenceContainer *m_psc;
};

/*!
 * \brief An adaptor to turn a const SequenceContainer like
 * std::vector or std::array into an Lvalue Property Map.
 * \tparam SequenceContainer A SequenceContainer is a Container that
 * stores objects of the same type in a linear arrangement.
 */
template <typename SequenceContainer>
class Index_property_map_const
  : public boost::put_get_helper<
    const typename SequenceContainer::value_type &,
    Index_property_map_const<SequenceContainer>> {
public:
  // typedefs
  typedef typename SequenceContainer::size_type key_type;
  typedef typename SequenceContainer::value_type value_type;
  typedef const value_type &reference;
  typedef typename boost::lvalue_property_map_tag category;

  Index_property_map_const() : m_psc(nullptr) {}

  Index_property_map_const(const SequenceContainer &sc) : m_psc(&sc) {}

  reference operator[](const key_type &k) const {
    return (*m_psc)[k];
  }

private:
  const SequenceContainer *m_psc;
};

} // namespace cm

#endif // CM_INDEX_PROPERTY_MAP_H
