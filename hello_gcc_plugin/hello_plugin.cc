#include <iostream>

#include "gcc-plugin.h"
#include "plugin-version.h"

int plugin_is_GPL_compatible;

extern void register_callbacks(const char *base_name);
int plugin_init(
    struct plugin_name_args *plugin_info, struct plugin_gcc_version *version) {
    if (!plugin_default_version_check(version, &gcc_version)) {
        std::cerr << "This GCC plugin is for version "
                  << GCCPLUGIN_VERSION_MAJOR << "." << GCCPLUGIN_VERSION_MINOR
                  << "\n";
        return 1;
    }

    std::cerr << "plugin info\n";
    std::cerr << "===========\n";
    std::cerr << "base name: " << plugin_info->base_name << "\n";
    std::cerr << "full name: " << plugin_info->full_name << "\n";
    std::cerr << "argc: " << plugin_info->argc << "\n";

    for (int i = 0; i < plugin_info->argc; i++) {
        std::cerr << plugin_info->argv[i].key << " = "
                  << plugin_info->argv[i].value << "\n";
    }
    if (plugin_info->version != NULL)
        std::cerr << "Version string of the plugin: " << plugin_info->version
                  << "\n";
    if (plugin_info->help != NULL)
        std::cerr << "Help string of the plugin: " << plugin_info->help << "\n";

    register_callbacks(plugin_info->base_name);
    return 0;
}
