#pragma once

#include <concepts>
#include <utility>

namespace venusaur {
template <typename Resource, Resource Invalid, typename Deleter>
    requires std::invocable<Deleter&, Resource>
class UniqueResource {
public:
    constexpr UniqueResource() noexcept = default;
    constexpr explicit UniqueResource(Resource resource) noexcept : m_resource(resource) {}

    ~UniqueResource() noexcept { reset(); }

    UniqueResource(const UniqueResource&) = delete;
    UniqueResource& operator=(const UniqueResource&) = delete;

    constexpr UniqueResource(UniqueResource&& other) noexcept
        : m_resource(std::exchange(other.m_resource, Invalid)), m_deleter(std::move(other.m_deleter)) {}

    constexpr UniqueResource& operator=(UniqueResource&& other) noexcept {
        if (this != &other) {
            reset();
            m_resource = std::exchange(other.m_resource, Invalid);
            m_deleter = std::move(other.m_deleter);
        }
        return *this;
    }

    [[nodiscard]] constexpr Resource get() const noexcept { return m_resource; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return m_resource != Invalid; }

    [[nodiscard]] constexpr Resource release() noexcept { return std::exchange(m_resource, Invalid); }

    constexpr void reset(Resource replacement = Invalid) noexcept {
        Resource previous = std::exchange(m_resource, replacement);
        if (previous != Invalid) {
            m_deleter(previous);
        }
    }

private:
    Resource m_resource = Invalid;
    Deleter m_deleter{};
};
} // namespace venusaur
