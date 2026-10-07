import SwiftUI

struct DubView: View {
    var body: some View {
        ContentUnavailableView("Dub", systemImage: "person.wave.2",
                               description: Text("Coming in a later build step."))
    }
}
